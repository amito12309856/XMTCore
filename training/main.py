
from __future__ import annotations

import gc
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor
import os
import random
import threading
import time
import traceback
from collections import deque

import chess
import numpy as np

MAX_PLIES = 260
AUTO_SAVE_SECONDS = 300
SIMULATION_WORKERS = 12

PIECE_VALUE = {
    chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3,
    chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0,
}


def canonical_square(square, turn):
    """Espelha verticalmente as casas quando o lado a jogar é Preto."""
    return square if turn == chess.WHITE else square ^ 56


def _is_game_over_fast(board):
    """Evita verificar repetição em posições nas quais ela ainda é impossível."""
    if board.is_game_over(claim_draw=False):
        return True

    return (
        board.halfmove_clock >= 8
        and board.can_claim_draw()
    )


def _position_payload(board, include_pieces):
    """Codifica pela perspectiva do lado a jogar; UI mantém casas reais."""
    piece_map = board.piece_map()
    encoded = np.zeros((12, 8, 8), dtype=np.float32)
    ui_pieces = {} if include_pieces else None

    for square, piece in piece_map.items():
        relative_square = canonical_square(square, board.turn)
        channel = piece.piece_type - 1
        if piece.color != board.turn:
            channel += 6

        row = 7 - chess.square_rank(relative_square)
        col = chess.square_file(relative_square)
        encoded[channel, row, col] = 1.0

        if include_pieces:
            ui_pieces[square] = (piece.piece_type, piece.color)

    return encoded, ui_pieces


def _board_material_balance(board):
    """Retorna a vantagem material das Brancas em pontos."""
    balance = 0

    for piece in board.piece_map().values():
        value = PIECE_VALUE.get(piece.piece_type, 0)
        balance += value if piece.color == chess.WHITE else -value

    return balance


def _fast_tactical_bonuses(board, legal_moves):
    """Pontuação tática ultrarrápida com incentivos de abertura."""
    mover = board.turn
    opp = not mover
    bonuses = []

    for move in legal_moves:
        attacker = board.piece_at(move.from_square)
        victim = board.piece_at(move.to_square)

        if board.is_en_passant(move):
            victim = chess.Piece(chess.PAWN, opp)

        is_castling = board.is_castling(move)
        is_attacked = board.is_attacked_by(opp, move.to_square)
        bonus = 0.0

        moving_value = PIECE_VALUE.get(
            move.promotion if move.promotion else (
                attacker.piece_type if attacker else chess.PAWN
            ),
            0,
        )
        victim_value = (
            PIECE_VALUE.get(victim.piece_type, 0)
            if victim else 0
        )

        # Evita passeios prematuros do Rei.
        if (
            attacker is not None
            and attacker.piece_type == chess.KING
            and not is_castling
            and board.fullmove_number <= 15
        ):
            bonus -= 2.5

        # Incentiva o roque.
        if is_castling:
            bonus += 1.2

        # Incentiva o desenvolvimento de Cavalos e Bispos.
        if (
            attacker is not None
            and attacker.piece_type in (chess.KNIGHT, chess.BISHOP)
            and board.fullmove_number <= 12
        ):
            home_rank = 0 if mover == chess.WHITE else 7

            if (
                chess.square_rank(move.from_square) == home_rank
                and chess.square_rank(move.to_square) != home_rank
            ):
                bonus += 0.35

        # Incentiva peões a ocuparem o centro e avançarem.
        if attacker is not None and attacker.piece_type == chess.PAWN:
            target_file = chess.square_file(move.to_square)
            target_rank = chess.square_rank(move.to_square)

            relative_rank = (
                target_rank
                if mover == chess.WHITE
                else 7 - target_rank
            )

            if (
                2 <= target_file <= 5
                or relative_rank >= 5
            ):
                bonus += 0.20

        if victim and not is_attacked:
            bonus += 0.6 * victim_value

        check_rep = board.halfmove_clock >= 2
        if victim or move.promotion or is_attacked or check_rep:
            board.push(move)
            try:
                if board.is_checkmate():
                    bonus += 12.0

                if check_rep and board.is_repetition(2):
                    bonus -= 4.5

                if is_attacked:
                    for sq in board.attackers(opp, move.to_square):
                        p = board.piece_at(sq)

                        if p is not None and p.piece_type in (
                            chess.PAWN,
                            chess.KNIGHT,
                            chess.BISHOP,
                        ):
                            if moving_value > victim_value:
                                bonus -= 0.5 * moving_value
                            else:
                                bonus += 0.25 * victim_value

                            break
            finally:
                board.pop()

        # Destrava jogos parados sem capturas ou avanço de peão
        if board.halfmove_clock >= 16 and (
            victim or (attacker is not None and attacker.piece_type == chess.PAWN)
        ):
            bonus += 0.40


        bonuses.append(float(bonus))

    return bonuses


class Match:
    """Partida mantida exclusivamente no processo que recebeu seu ID."""

    __slots__ = (
        "id", "genome", "board", "plies", "fitness", "material",
        "checks", "status", "training_samples", "last_action",
        "mate_window", "material_lead_side",
        "material_lead_streak", "adjudication_result",
    )

    def __init__(self, match_id, genome):
        self.id = int(match_id)
        self.genome = int(genome)
        self.board = chess.Board()
        self.plies = 0
        self.fitness = 0.0
        self.material = 0.0
        self.checks = 0
        self.status = "ativo"
        self.training_samples = []
        self.last_action = 0
        self.mate_window = deque(maxlen=15)

        self.material_lead_side = None
        self.material_lead_streak = 0
        self.adjudication_result = None

    def step(self, move):
        board = self.board
        mover = board.turn

        victim = board.piece_at(move.to_square)
        if board.is_en_passant(move):
            victim = chess.Piece(chess.PAWN, not mover)

        capture_value = PIECE_VALUE.get(victim.piece_type, 0) if victim else 0

        if victim:
            self.material += capture_value if mover == chess.WHITE else -capture_value
            self.fitness += 0.35 * capture_value

        from_square = canonical_square(move.from_square, mover)
        to_square = canonical_square(move.to_square, mover)
        action = from_square * 64 + to_square

        encoded, _ = _position_payload(board, False)
        self.mate_window.append((encoded, action, bool(mover), 3.0))

        if len(self.training_samples) < 12 and random.random() < 0.12:
            weight = 1.0 + 0.5 * min(capture_value, 9) / 9.0
            self.training_samples.append((encoded, action, bool(mover), weight))

        board.push(move)

        # Penaliza imediatamente a repetição de uma posição.
        if board.is_repetition(2):
            self.fitness -= 3.0

        self.last_action = action

        if move.promotion:
            self.fitness += 1.5

        self.plies += 1

        if board.is_check():
            if self.checks < 3:
                self.fitness += 0.5
            self.checks += 1

        # Adjudicação de vantagem material esmagadora após 80 meios-lances.
        # Adjudicação de vantagem material (+3) após 60 meios-lances.
        if self.plies >= 60:
            material_balance = _board_material_balance(board)

            if material_balance >= 3:
                leading_side = chess.WHITE
            elif material_balance <= -3:
                leading_side = chess.BLACK
            else:
                leading_side = None

            if leading_side is None:
                self.material_lead_side = None
                self.material_lead_streak = 0

            elif leading_side == self.material_lead_side:
                self.material_lead_streak += 1

            else:
                self.material_lead_side = leading_side
                self.material_lead_streak = 1

            if self.material_lead_streak >= 8:

                self.adjudication_result = (
                    "1-0"
                    if self.material_lead_side == chess.WHITE
                    else "0-1"
                )
                self.finish()

    def finish(self):
        result = (
            self.adjudication_result
            if self.adjudication_result is not None
            else (
                self.board.result(claim_draw=True)
                if _is_game_over_fast(self.board)
                else "1/2-1/2"
            )
        )

        if self.board.is_checkmate():
            self.training_samples.extend(
                (state, action, mover, 3.0, None, "mate")
                for state, action, mover, _ in self.mate_window
            )

        if result in ("1-0", "0-1"):
            self.fitness += 30.0 + max(0, MAX_PLIES - self.plies) * 0.03
        else:
            self.fitness -= 10.0

        if abs(float(self.material)) >= 5.0:
            encoded, _ = _position_payload(self.board, False)

            material_side = (
                self.material
                if self.board.turn == chess.WHITE
                else -self.material
            )
            material_target = float(np.tanh(material_side / 6.0))

            self.training_samples.append((
                encoded, 0, bool(self.board.turn),
                1.25, material_target, "material",
            ))

        self.status = result


def _collect_process_state(
    matches,
    reported,
    include_pieces,
    focus_id,
    turbo=False,
):
    states = []
    match_ids = []
    genome_ids = []
    legal_moves = []
    tactical_bonuses = []
    plies = []
    turns = []
    fens = [] if not turbo else None
    completed = []
    ui_boards = []
    focus_fen = None

    for match in matches.values():
        board = match.board

        if match.status == "ativo":
            if _is_game_over_fast(board):
                match.finish()
            else:
                legal = list(board.legal_moves)

                if not legal:
                    match.finish()
                else:
                    encoded, ui_pieces = _position_payload(
                        board,
                        include_pieces,
                    )

                    states.append(encoded)
                    match_ids.append(match.id)
                    genome_ids.append(match.genome)
                    plies.append(match.plies)
                    turns.append(board.turn)

                    legal_moves.append([
                        (
                            move.from_square,
                            move.to_square,
                            move.promotion or 0,
                        )
                        for move in legal
                    ])

                    tactical_bonuses.append(
                        _fast_tactical_bonuses(board, legal)
                    )

                    if fens is not None:
                        fens.append(board.fen())

                    if include_pieces:
                        ui_boards.append({
                            "id": match.id,
                            "genome": match.genome,
                            "pieces": ui_pieces,
                            "status": match.status,
                        })

                    if match.id == focus_id:
                        focus_fen = board.fen()

                    continue

        if include_pieces:
            _, ui_pieces = _position_payload(board, True)

            ui_boards.append({
                "id": match.id,
                "genome": match.genome,
                "pieces": ui_pieces,
                "status": match.status,
            })

            if match.id == focus_id:
                focus_fen = board.fen()

        if match.status != "ativo" and match.id not in reported:
            completed.append({
                "id": match.id,
                "genome": match.genome,
                "fitness": float(match.fitness),
                "status": match.status,
                "samples": list(match.training_samples),
            })
            reported.add(match.id)

    state_array = (
        np.stack(states)
        if states
        else np.empty((0, 12, 8, 8), dtype=np.float32)
    )

    return {
        "states": state_array,
        "ids": match_ids,
        "genomes": genome_ids,
        "legal_moves": legal_moves,
        "tactical_bonuses": tactical_bonuses,
        "plies": plies,
        "turns": turns,
        "fens": fens,
        "completed": completed,
        "ui_boards": ui_boards,
        "focus_fen": focus_fen,
    }


def _simulation_process_main(connection):
    """Loop persistente: cada processo cria, atualiza e mantém seus próprios Boards."""
    matches = {}
    reported = set()

    try:
        while True:
            command, payload = connection.recv()

            if command == "shutdown":
                break

            if command == "start":
                specs, include_pieces, focus_id, turbo = payload
                matches = {
                    match_id: Match(match_id, genome)
                    for match_id, genome in specs
                }
                reported = set()

                response = _collect_process_state(
                    matches,
                    reported,
                    include_pieces,
                    focus_id,
                    turbo,
                )
                connection.send(("ok", response))

            elif command == "step":
                moves_by_id, include_pieces, focus_id, turbo = payload

                for match_id, move_data in moves_by_id.items():
                    match = matches.get(match_id)

                    if match is None or match.status != "ativo":
                        continue

                    if move_data is None:
                        match.finish()
                        continue

                    from_square, to_square, promotion = move_data
                    move = chess.Move(
                        int(from_square),
                        int(to_square),
                        promotion=int(promotion) if promotion else None,
                    )
                    match.step(move)

                response = _collect_process_state(
                    matches,
                    reported,
                    include_pieces,
                    focus_id,
                    turbo,
                )
                connection.send(("ok", response))

            elif command == "finalize":
                include_pieces, focus_id, turbo = payload

                for match in matches.values():
                    if match.status == "ativo":
                        match.finish()

                response = _collect_process_state(
                    matches,
                    reported,
                    include_pieces,
                    focus_id,
                    turbo,
                )
                connection.send(("ok", response))

            else:
                raise ValueError(f"Comando desconhecido do worker: {command}")

    except EOFError:
        pass
    except BaseException:
        try:
            connection.send(("error", traceback.format_exc()))
        except Exception:
            pass
    finally:
        connection.close()


class SimulationPool:
    """Pool persistente com afinidade fixa: um subprocesso é dono de cada grupo de jogos."""

    def __init__(self, workers=SIMULATION_WORKERS):
        self.worker_count = max(
            1, min(int(workers), os.cpu_count() or int(workers))
        )
        self.context = mp.get_context("spawn")
        self.processes = []
        self.connections = []
        self.assignments = []
        self.id_to_worker = {}

        for index in range(self.worker_count):
            parent_conn, child_conn = self.context.Pipe(duplex=True)
            process = self.context.Process(
                target=_simulation_process_main,
                args=(child_conn,),
                name=f"ChessSim-{index}",
                daemon=True,
            )
            process.start()
            child_conn.close()

            self.connections.append(parent_conn)
            self.processes.append(process)

    def _request_all(self, requests):
        for connection, request in zip(self.connections, requests):
            connection.send(request)

        responses = []

        for connection in self.connections:
            try:
                status, payload = connection.recv()
            except (EOFError, OSError) as exc:
                raise RuntimeError(
                    "Um processo de simulação encerrou inesperadamente. "
                    "Verifique o processo filho e o traceback no terminal."
                ) from exc

            if status != "ok":
                raise RuntimeError(
                    f"Falha em processo de simulação:\n{payload}"
                )

            responses.append(payload)

        return self._merge(responses)

    @staticmethod
    def _merge(responses):
        state_parts = [
            result["states"]
            for result in responses
            if len(result["states"])
        ]

        states = (
            np.concatenate(state_parts, axis=0)
            if state_parts
            else np.empty((0, 12, 8, 8), dtype=np.float32)
        )

        # No Turbo, todos os processos retornam fens=None.
        # No Watch, todos retornam suas listas de FENs.
        fens = (
            None
            if any(result.get("fens") is None for result in responses)
            else [
                item
                for result in responses
                for item in result["fens"]
            ]
        )

        return {
            "states": states,
            "ids": [
                item for result in responses for item in result["ids"]
            ],
            "genomes": [
                item for result in responses for item in result["genomes"]
            ],
            "legal_moves": [
                item
                for result in responses
                for item in result["legal_moves"]
            ],
            "tactical_bonuses": [
                item
                for result in responses
                for item in result.get("tactical_bonuses", [])
            ],
            "plies": [
                item
                for result in responses
                for item in result.get("plies", [])
            ],
            "turns": [
                item
                for result in responses
                for item in result.get("turns", [])
            ],
            "fens": fens,
            "completed": [
                item
                for result in responses
                for item in result["completed"]
            ],
            "ui_boards": [
                item
                for result in responses
                for item in result["ui_boards"]
            ],
            "focus_fen": next(
                (
                    result["focus_fen"]
                    for result in responses
                    if result["focus_fen"] is not None
                ),
                None,
            ),
        }

    def start_generation(self, specs, focus_id, turbo=False):
        chunks = [
            specs[index::self.worker_count]
            for index in range(self.worker_count)
        ]
        self.assignments = chunks

        self.id_to_worker = {
            match_id: worker_index
            for worker_index, chunk in enumerate(chunks)
            for match_id, _genome in chunk
        }

        requests = [
            ("start", (chunk, True, focus_id, turbo))
            for chunk in chunks
        ]
        return self._request_all(requests)

    def step(self, moves_by_id, include_pieces, focus_id, turbo=False):
        requests = []

        for chunk in self.assignments:
            local_moves = {
                match_id: moves_by_id[match_id]
                for match_id, _genome in chunk
                if match_id in moves_by_id
            }
            requests.append(
                ("step", (local_moves, include_pieces, focus_id, turbo))
            )

        return self._request_all(requests)

    def finalize(self, focus_id, turbo=False):
        requests = [
            ("finalize", (True, focus_id, turbo))
            for _ in self.connections
        ]
        return self._request_all(requests)

    def close(self):
        for connection in self.connections:
            try:
                connection.send(("shutdown", None))
            except (BrokenPipeError, EOFError, OSError):
                pass

        for process in self.processes:
            process.join(timeout=2.0)

            if process.is_alive():
                process.terminate()
                process.join(timeout=1.0)

        for connection in self.connections:
            try:
                connection.close()
            except OSError:
                pass


class Worker(threading.Thread):
    def __init__(self, brain):
        super().__init__(daemon=True)
        self.brain = brain
        self.stop_event = threading.Event()
        self.turbo = False
        self.count = 25
        self.lock = threading.Lock()

        self._benchmark_executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="ChessBenchmark",
        )
        self._scheduled_benchmarks = set()
        self._benchmark_schedule_lock = threading.Lock()

        start_board = chess.Board()
        start_pieces = {
            square: (piece.piece_type, piece.color)
            for square, piece in start_board.piece_map().items()
        }

        self.snapshot = {
            "fen": chess.STARTING_FEN,
            "focus_pieces": start_pieces,
            "boards": [],
            "generation": brain.generation,
            "rating": brain.rating,
            "device": brain.device_name,
            "loss": brain.last_loss,
            "evaluation": 0.0,
            "elo_history": list(brain.elo_history),
            "focus_id": 1,
            "turbo": False,
            "result_totals": dict(brain.result_totals),
            "recent_results": list(brain.recent_results),
            "pending_game_counts": list(brain.pending_game_counts),
            "last_benchmark_generation": brain.last_benchmark_generation,
        }

        self.best_fitness = 0.0
        self.focus_id = 1
        self.ui_boards = {}
        self.focus_fen = chess.STARTING_FEN

    def stop(self):
        self.stop_event.set()

    def set_count(self, count):
        self.count = max(1, min(100, int(count)))

    def set_focus(self, match_id):
        if match_id is not None:
            self.focus_id = int(match_id)

    def _consume_response(self, response):
        for item in response["ui_boards"]:
            self.ui_boards[item["id"]] = item

        if response["focus_fen"] is not None:
            self.focus_fen = response["focus_fen"]

    def _record_completed(self, response):
        completed = response.get("completed", [])

        if completed:
            self.brain.record_completed_matches(completed)

        return len(completed)

    def _benchmark_task(self, model, generation):
        try:
            from brain import run_reference_benchmark

            distillation_samples = []
            results = run_reference_benchmark(
                model,
                move_delay=0.002,
                training_samples=distillation_samples,
            )

            if distillation_samples:
                self.brain.record_benchmark_samples(distillation_samples)

            self.brain.apply_benchmark_results(results, generation)
            self.brain.save_checkpoint()

        except Exception:
            print(
                f"[ERRO] Benchmark da geração {generation}:\n"
                + traceback.format_exc()
            )

    def _schedule_benchmark(self, generation):
        generation = int(generation)

        if generation < 5 or generation % 5 != 0:
            return

        with self._benchmark_schedule_lock:
            if generation <= self.brain.last_benchmark_generation:
                return

            if generation in self._scheduled_benchmarks:
                return

            model = self.brain.make_benchmark_model()
            self._scheduled_benchmarks.add(generation)
            self._benchmark_executor.submit(
                self._benchmark_task, model, generation
            )

    def _schedule_missed_benchmark(self):
        latest_due = (self.brain.generation // 5) * 5

        if (
            latest_due >= 5
            and latest_due > self.brain.last_benchmark_generation
        ):
            self._schedule_benchmark(latest_due)

    def publish(self, value=0.0):
        boards = [
            self.ui_boards[key]
            for key in sorted(self.ui_boards)
        ]

        focus = self.ui_boards.get(self.focus_id)

        if focus is None and boards:
            focus = boards[0]

        snapshot = {
            "fen": self.focus_fen,
            "focus_pieces": focus["pieces"] if focus else {},
            "boards": boards,
            "generation": self.brain.generation,
            "rating": self.brain.rating,
            "device": self.brain.device_name,
            "elo_history": list(self.brain.elo_history),
            "focus_id": self.focus_id,
            "turbo": self.turbo,
            "loss": float(self.brain.last_loss),
            "evaluation": float(value),
            "best_fitness": self.best_fitness,
            "result_totals": dict(self.brain.result_totals),
            "recent_results": list(self.brain.recent_results),
            "pending_game_counts": list(self.brain.pending_game_counts),
            "last_benchmark_generation": self.brain.last_benchmark_generation,
        }

        with self.lock:
            self.snapshot = snapshot

    def get_snapshot(self):
        with self.lock:
            return self.snapshot.copy()

    def run(self):
        pool = None

        try:
            pool = SimulationPool(SIMULATION_WORKERS)
            self._schedule_missed_benchmark()

            while not self.stop_event.is_set():
                match_count = self.count
                start_genome = self.brain.next_genome_index

                specs = [
                    (
                        match_id,
                        (start_genome + match_id - 1) % 20,
                    )
                    for match_id in range(1, match_count + 1)
                ]

                self.ui_boards = {}
                self.focus_fen = chess.STARTING_FEN
                completed_this_round = 0

                batch = pool.start_generation(
                    specs,
                    self.focus_id,
                    self.turbo,
                )
                self._consume_response(batch)
                completed_this_round += self._record_completed(batch)

                self.publish()
                last_publish = time.monotonic()
                focus_value = 0.0

                for _ply in range(MAX_PLIES):
                    if self.stop_event.is_set() or not batch["ids"]:
                        break

                    moves, values = self.brain.batch_choose_encoded(
                        batch["states"],
                        batch["genomes"],
                        batch["legal_moves"],
                        temperature=0.55 if self.turbo else 0.35,
                        fens=batch.get("fens"),
                        batch_size=match_count,
                        tactical_bonuses=batch.get("tactical_bonuses"),
                        turbo=self.turbo,
                        plies=batch.get("plies"),
                        turns=batch.get("turns"),
                    )

                    value_by_id = dict(zip(batch["ids"], values))
                    focus_value = value_by_id.get(
                        self.focus_id, focus_value
                    )
                    move_by_id = dict(zip(batch["ids"], moves))

                    now = time.monotonic()
                    pub_interval = 0.35 if len(batch["ids"]) <= 15 else 0.15
                    include_pieces = (
                        not self.turbo or now - last_publish >= pub_interval
                    )

                    batch = pool.step(
                        move_by_id,
                        include_pieces,
                        self.focus_id,
                        self.turbo,
                    )

                    completed_this_round += self._record_completed(batch)
                    self._consume_response(batch)

                    if include_pieces:
                        self.publish(focus_value)
                        last_publish = time.monotonic()

                    if not self.turbo:
                        self.stop_event.wait(0.012)

                if self.stop_event.is_set():
                    break

                final_batch = pool.finalize(
                    self.focus_id,
                    self.turbo,
                )
                completed_this_round += self._record_completed(final_batch)
                self._consume_response(final_batch)
                self.publish(focus_value)

                self.brain.next_genome_index = (
                    start_genome + match_count
                ) % 20

                if self.brain.training_cycle_ready():
                    scores = [
                        self.brain.pending_score_totals[index]
                        / self.brain.pending_game_counts[index]
                        for index in range(20)
                    ]

                    self.best_fitness = max(scores)
                    self.brain.evolve(scores)
                    gradient_loss = self.brain.last_loss

                    print(
                        f"[GERAÇÃO {self.brain.generation}] ciclo completo | "
                        f"{self.brain.games_trained} partidas acumuladas | "
                        f"fitness={self.best_fitness:.2f} | "
                        f"loss={gradient_loss:.5f} | "
                        f"rating real={self.brain.rating:.1f}"
                    )

                    self._schedule_benchmark(self.brain.generation)

                else:
                    covered = sum(
                        1
                        for number in self.brain.pending_game_counts
                        if number > 0
                    )

                    print(
                        f"[ACÚMULO] +{completed_this_round} partidas | "
                        f"{covered}/20 indivíduos já jogaram | "
                        f"geração permanece {self.brain.generation} | "
                        f"rating={self.brain.rating:.1f}"
                    )

                self.publish(focus_value)

                if self.brain.should_auto_save(AUTO_SAVE_SECONDS):
                    self.brain.save_checkpoint()

                del specs
                gc.collect()

        except Exception:
            print(
                "[ERRO] Worker principal:\n"
                + traceback.format_exc()
            )
            self.stop_event.set()

        finally:
            if pool is not None:
                pool.close()

            self._benchmark_executor.shutdown(
                wait=True,
                cancel_futures=False,
            )


def main():
    import pygame
    from brain import ChessBrain
    from ui import ChessUI

    brain = ChessBrain()

    print(
        f"[INFO] {brain.device_name} | "
        f"partidas={brain.games_trained} | "
        f"geração={brain.generation} | "
        f"população=20 | workers={SIMULATION_WORKERS}"
    )

    worker = Worker(brain)
    ui = ChessUI()
    worker.start()

    running = True

    try:
        while running:
            quit_, keys, changed, focus_id = ui.events()

            if quit_:
                running = False

            if changed is not None:
                worker.set_count(changed)

            if focus_id is not None:
                worker.set_focus(focus_id)

            for key in keys:
                if key == pygame.K_ESCAPE:
                    running = False
                elif key == pygame.K_SPACE:
                    worker.turbo = not worker.turbo
                elif key == pygame.K_s:
                    brain.save_checkpoint()

            snapshot = worker.get_snapshot()
            ui.draw(
                snapshot.get("focus_pieces", {}),
                snapshot,
            )

    finally:
        worker.stop()
        worker.join()
        brain.save_checkpoint()
        ui.close()


if __name__ == "__main__":
    mp.freeze_support()
    main()
