from __future__ import annotations

import os
import random
import time
import threading
from collections import OrderedDict, deque
from pathlib import Path

import numpy as np
import chess
import torch
import torch.nn as nn
import torch.nn.functional as F


CHECKPOINT_PATH = Path("checkpoint.pth")
POPULATION_SIZE = 20
MAX_CHECKPOINT_BYTES = 15 * 1024 * 1024

CHECKPOINT_FORMAT = "neural-chess-canonical-v3"

RATING_METHOD = "reference-benchmark-v1"
ELO_K = 12.0

REFERENCE_BOTS = {
    "novice": 400.0,                 # Iniciante
    "casual": 750.0,                 # Jogador Casual
    "intermediate": 1050.0,          # Intermediário
    "tournament": 1350.0,            # Jogador de Torneio
    "club_player": 1650.0,           # Jogador de Clube
    "candidate_master": 1950.0,      # Candidato a Mestre (CM)
    "fide_master": 2300.0,           # Mestre FIDE (FM)
    "international_master": 2450.0,  # Mestre Internacional (IM)
}

BENCHMARK_GAMES_PER_BOT = 4
BENCHMARK_MAX_PLIES = 220

PROMO_INDEX = {
    None: 0,
    chess.KNIGHT: 1,
    chess.BISHOP: 2,
    chess.ROOK: 3,
    chess.QUEEN: 4,
}

PIECE_VALUE = {
    chess.PAWN: 1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK: 5,
    chess.QUEEN: 9,
    chess.KING: 0,
}

VALUE_HEAD_WEIGHT = 0.35


def canonical_square(square, turn):
    """Espelha verticalmente as casas quando o lado a jogar é Preto."""
    return square if turn == chess.WHITE else square ^ 56


def _encode_board_array(board):
    """Codificação canônica: lado a jogar nos canais 0-5."""
    arr = np.zeros((12, 8, 8), dtype=np.float32)

    for square, piece in board.piece_map().items():
        relative_square = canonical_square(square, board.turn)

        channel = piece.piece_type - 1
        if piece.color != board.turn:
            channel += 6

        row = 7 - chess.square_rank(relative_square)
        col = chess.square_file(relative_square)

        arr[channel, row, col] = 1.0

    return arr


def encode_board(board):
    return torch.from_numpy(_encode_board_array(board))


def move_to_action(move, turn=chess.WHITE):
    """Converte um lance real para o índice da orientação canônica."""
    from_square = canonical_square(move.from_square, turn)
    to_square = canonical_square(move.to_square, turn)
    return from_square * 64 + to_square


class PolicyValueNet(nn.Module):
    """Rede compacta com Policy Head e Value Head."""

    def __init__(self, channels=32, embedding=16):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(12, channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.ReLU(inplace=True),
        )

        self.policy_features = nn.Conv2d(channels, embedding, 1)
        self.from_head = nn.Linear(embedding, embedding, bias=False)
        self.to_head = nn.Linear(embedding, embedding, bias=False)
        self.promotion_bias = nn.Parameter(torch.zeros(5))

        self.value_head = nn.Sequential(
            nn.Linear(channels, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
            nn.Tanh(),
        )

    def forward(self, x, value_only=False):
        features = self.features(x)

        value = self.value_head(
            features.mean(dim=(2, 3))
        ).squeeze(-1)

        if value_only:
            return value

        squares = (
            self.policy_features(features)
            .flatten(2)
            .transpose(1, 2)
        )

        src = self.from_head(squares)
        dst = self.to_head(squares)

        policy = torch.matmul(
            src, dst.transpose(1, 2)
        ) / (src.shape[-1] ** 0.5)

        return policy.flatten(1), value

    def score_legal_moves(self, logits, moves, turn=chess.WHITE):
        actions = torch.tensor(
            [move_to_action(move, turn) for move in moves],
            device=logits.device,
            dtype=torch.long,
        )

        promotions = torch.tensor(
            [PROMO_INDEX.get(move.promotion, 0) for move in moves],
            device=logits.device,
            dtype=torch.long,
        )

        return logits[actions] + self.promotion_bias[promotions]


def _move_policy_score(logits, promotion_bias, move, turn=chess.WHITE):
    action = move_to_action(move, turn)
    promo = PROMO_INDEX.get(move.promotion, 0)

    return (
        float(logits[action])
        + float(promotion_bias[promo])
    )


def _forcing_moves(board, legal_moves):
    forcing = []

    for move in legal_moves:
        victim = board.piece_at(move.to_square)

        if board.is_en_passant(move):
            capture_value = PIECE_VALUE[chess.PAWN]
        else:
            capture_value = (
                PIECE_VALUE.get(victim.piece_type, 0)
                if victim else 0
            )

        if capture_value >= 3 or move.promotion:
            forcing.append(move)

    return forcing


def _candidate_moves(
    board,
    legal_moves,
    logits,
    promotion_bias,
    top_n=5,
):
    ranked = sorted(
        legal_moves,
        key=lambda move: _move_policy_score(
            logits, promotion_bias, move, board.turn
        ),
        reverse=True,
    )

    selected = ranked[:top_n]
    seen = set(selected)

    for move in _forcing_moves(board, legal_moves):
        if move not in seen:
            selected.append(move)
            seen.add(move)

    return selected


def _board_ply(board):
    return max(
        0,
        (int(board.fullmove_number) - 1) * 2
        + (0 if board.turn else 1),
    )


def _fen_ply(fen):
    fields = fen.split()

    if len(fields) < 6:
        return 0

    try:
        fullmove = int(fields[5])
    except (TypeError, ValueError):
        return 0

    return max(
        0,
        (fullmove - 1) * 2
        + (0 if fields[1] == "w" else 1),
    )


def _choose_policy_move(
    move_tuples,
    logits,
    promotion_bias,
    temperature,
    tactical_bonuses=None,
    turn=chess.WHITE,
    dirichlet_noise=False,
):
    if not move_tuples:
        return None

    scores = np.empty(len(move_tuples), dtype=np.float64)
    normalized = []

    for index, item in enumerate(move_tuples):
        frm, to, promotion = item
        frm = int(frm)
        to = int(to)
        promotion = int(promotion or 0)

        canonical_from = canonical_square(frm, turn)
        canonical_to = canonical_square(to, turn)
        action = canonical_from * 64 + canonical_to

        promo_index = PROMO_INDEX.get(
            promotion if promotion else None,
            0,
        )

        scores[index] = (
            float(logits[action])
            + float(promotion_bias[promo_index])
        )

        if tactical_bonuses is not None and index < len(tactical_bonuses):
            scores[index] += float(tactical_bonuses[index])

        normalized.append((frm, to, promotion))

    if temperature <= 0:
        return normalized[int(np.argmax(scores))]

    scaled = (
        scores - scores.max()
    ) / max(0.02, float(temperature))

    probabilities = np.exp(np.clip(scaled, -60.0, 0.0))
    probabilities /= probabilities.sum()

    if dirichlet_noise and len(probabilities) > 1:
        noise = np.random.dirichlet([0.3] * len(probabilities))
        probabilities = 0.75 * probabilities + 0.25 * noise
        probabilities /= probabilities.sum()

    return normalized[
        int(np.random.choice(len(normalized), p=probabilities))
    ]


def _choose_top_k_policy_move(
    move_tuples,
    logits,
    promotion_bias,
    top_k=4,
    temperature=1.2,
    tactical_bonuses=None,
    turn=chess.WHITE,
):
    if not move_tuples:
        return None

    scored = []

    for index, item in enumerate(move_tuples):
        frm, to, promotion = item
        frm = int(frm)
        to = int(to)
        promotion = int(promotion or 0)

        canonical_from = canonical_square(frm, turn)
        canonical_to = canonical_square(to, turn)
        action = canonical_from * 64 + canonical_to

        promo_index = PROMO_INDEX.get(
            promotion if promotion else None,
            0,
        )

        score = (
            float(logits[action])
            + float(promotion_bias[promo_index])
        )

        if tactical_bonuses is not None and index < len(tactical_bonuses):
            score += float(tactical_bonuses[index])

        scored.append(((frm, to, promotion), score))

    scored.sort(key=lambda pair: pair[1], reverse=True)

    selected = scored[
        :max(1, min(int(top_k), len(scored)))
    ]

    values = np.asarray(
        [item[1] for item in selected],
        dtype=np.float64,
    )

    values = np.maximum(values, values.max() - 2.0)

    scaled = (
        values - values.max()
    ) / max(0.02, float(temperature))

    probabilities = np.exp(np.clip(scaled, -60.0, 0.0))
    probabilities /= probabilities.sum()

    selected_index = int(
        np.random.choice(len(selected), p=probabilities)
    )

    return selected[selected_index][0]

def _generate_endgame_drills(count=64):
    """
    Gera posições legais de finais ganhos:
    Rei + Dama vs Rei, Rei + Torre vs Rei
    e Rei + duas Torres vs Rei.
    """
    samples = []

    configurations = (
        (chess.QUEEN,),
        (chess.ROOK,),
        (chess.ROOK, chess.ROOK),
    )

    for _ in range(max(0, int(count))):
        accepted = False

        for _attempt in range(200):
            board = chess.Board(None)

            squares = random.sample(range(64), 3 + random.choice((0, 1)))
            white_king, black_king = squares[:2]

            if chess.square_distance(
                white_king,
                black_king,
            ) <= 1:
                continue

            configuration = random.choice(configurations)

            if len(squares) < 2 + len(configuration):
                continue

            board.set_piece_at(
                white_king,
                chess.Piece(chess.KING, chess.WHITE),
            )

            board.set_piece_at(
                black_king,
                chess.Piece(chess.KING, chess.BLACK),
            )

            for square, piece_type in zip(
                squares[2:],
                configuration,
            ):
                board.set_piece_at(
                    square,
                    chess.Piece(piece_type, chess.WHITE),
                )

            board.turn = chess.WHITE

            if not board.is_valid():
                continue

            legal_moves = list(board.legal_moves)

            if not legal_moves:
                continue

            best_move = None
            best_score = -float("inf")

            old_king_distance = chess.square_distance(
                white_king,
                black_king,
            )

            for move in legal_moves:
                board.push(move)

                try:
                    if board.is_checkmate():
                        score = 100000.0

                    elif board.is_stalemate():
                        score = -100000.0

                    else:
                        score = 0.0

                        if board.is_check():
                            score += 100.0

                        new_white_king = board.king(chess.WHITE)
                        new_black_king = board.king(chess.BLACK)

                        new_king_distance = chess.square_distance(
                            new_white_king,
                            new_black_king,
                        )

                        score += (
                            old_king_distance - new_king_distance
                        ) * 4.0

                        moved_piece = board.piece_at(move.to_square)

                        if (
                            moved_piece is not None
                            and moved_piece.piece_type != chess.KING
                        ):
                            piece_distance = chess.square_distance(
                                move.to_square,
                                new_black_king,
                            )

                            score -= piece_distance * 0.5

                        score += random.random() * 0.01

                finally:
                    board.pop()

                if score > best_score:
                    best_score = score
                    best_move = move

            if best_move is None:
                continue

            samples.append((
                _encode_board_array(board),
                move_to_action(best_move, chess.WHITE),
                1.0,
                1.5,
                1.0,
            ))

            accepted = True
            break

        if not accepted:
            continue

    return samples

def _material_score(board, perspective):
    balance = 0.0

    for piece in board.piece_map().values():
        value = PIECE_VALUE.get(piece.piece_type, 0)
        balance += value if piece.color == perspective else -value

    mopup_bonus = 0.0
    if abs(balance) >= 3.0:
        winner = perspective if balance > 0 else not perspective
        loser = not winner
        win_king = board.king(winner)
        lose_king = board.king(loser)
        if win_king is not None and lose_king is not None:
            # Empurra o rei perdedor para a borda/canto do tabuleiro
            lose_file = chess.square_file(lose_king)
            lose_rank = chess.square_rank(lose_king)
            center_dist = abs(lose_file - 3.5) + abs(lose_rank - 3.5)
            # Aproxima o rei vencedor do rei perdedor para dar mate
            king_dist = chess.square_distance(win_king, lose_king)
            raw_mopup = 0.25 * center_dist + 0.12 * (14 - king_dist)
            mopup_bonus = raw_mopup if balance > 0 else -raw_mopup

    return float(np.tanh((balance + mopup_bonus) / 12.0))


def _terminal_score(board, perspective):
    if not board.is_checkmate():
        return None

    winner = not board.turn

    return 2.0 if winner == perspective else -2.0


def _leaf_score(
    value_leaf,
    material_value,
    root_color,
    leaf_turn,
    value_weight=None,
):
    """Converte a Value Head da perspectiva da folha para a raiz."""
    value_root = (
        float(value_leaf)
        if leaf_turn == root_color
        else -float(value_leaf)
    )

    weight = (
        VALUE_HEAD_WEIGHT
        if value_weight is None
        else float(value_weight)
    )

    return (
        weight * value_root
        + (1.0 - weight) * float(material_value)
    )

_FAST_PIECE_VALS = (
    (chess.PAWN, 1.0),
    (chess.KNIGHT, 3.2),
    (chess.BISHOP, 3.35),
    (chess.ROOK, 5.0),
    (chess.QUEEN, 9.2),
)

_CENTER_MASK = (
    (1 << chess.D4) | (1 << chess.E4) | (1 << chess.D5) | (1 << chess.E5)
)
_EXT_CENTER_MASK = (
    _CENTER_MASK

    | (1 << chess.C3) | (1 << chess.D3) | (1 << chess.E3) | (1 << chess.F3)
    | (1 << chess.C4) | (1 << chess.F4) | (1 << chess.C5) | (1 << chess.F5)
    | (1 << chess.C6) | (1 << chess.D6) | (1 << chess.E6) | (1 << chess.F6)
)
_BACK_RANK_W = (1 << chess.B1) | (1 << chess.C1) | (1 << chess.F1) | (1 << chess.G1)
_BACK_RANK_B = (1 << chess.B8) | (1 << chess.C8) | (1 << chess.F8) | (1 << chess.G8)


def _fast_bitboard_eval(pos: chess.Board, root_color: bool) -> float:
    """Avaliação tática/posicional em Bitboards nativos (11x mais rápida que loop de 64 casas)."""
    w_occ = pos.occupied_co[chess.WHITE]
    b_occ = pos.occupied_co[chess.BLACK]
    score_w = 0.0

    for p_type, val in _FAST_PIECE_VALS:
        p_mask = pos.pieces_mask(p_type, chess.WHITE)
        n_w = p_mask.bit_count()
        p_mask_b = pos.pieces_mask(p_type, chess.BLACK)
        n_b = p_mask_b.bit_count()
        score_w += val * (n_w - n_b)

        if p_type == chess.BISHOP:
            if n_w >= 2:
                score_w += 0.45
            if n_b >= 2:
                score_w -= 0.45

    # Controle do centro em operação bitwise pura
    score_w += 0.14 * ((w_occ & _CENTER_MASK).bit_count() - (b_occ & _CENTER_MASK).bit_count())
    score_w += 0.06 * ((w_occ & _EXT_CENTER_MASK).bit_count() - (b_occ & _EXT_CENTER_MASK).bit_count())

    # Penalidade por peças menores ainda presas na primeira fileira no meio-jogo
    minors_w = pos.knights & w_occ | pos.bishops & w_occ
    minors_b = pos.knights & b_occ | pos.bishops & b_occ
    score_w -= 0.22 * (minors_w & _BACK_RANK_W).bit_count()
    score_w += 0.22 * (minors_b & _BACK_RANK_B).bit_count()

    # Segurança do rei (roque / canto vs centro exposto)
    wk = pos.king(chess.WHITE)
    bk = pos.king(chess.BLACK)
    if wk is not None:
        if wk in (chess.G1, chess.H1, chess.B1, chess.C1):
            score_w += 0.35
        elif wk in (chess.D1, chess.E1, chess.D2, chess.E2):
            score_w -= 0.25
    if bk is not None:
        if bk in (chess.G8, chess.H8, chess.B8, chess.C8):
            score_w -= 0.35
        elif bk in (chess.D8, chess.E8, chess.D7, chess.E7):
            score_w += 0.25

    return score_w if root_color == chess.WHITE else -score_w


@torch.inference_mode()
def _model_move(model, board, temperature=0.0, search_depth=5, time_limit=None):
    """Negamax exato com TT Flags, Quiescence, Check Extension e Ordenação Neural."""
    legal = list(board.legal_moves)
    if not legal:
        return None
    if len(legal) == 1:
        return next(iter(legal))

    start_time = time.perf_counter()
    deadline = (start_time + float(time_limit)) if time_limit is not None else None

    model.eval()
    device = next(model.parameters()).device
    root_color = board.turn
    root_state = encode_board(board).unsqueeze(0).to(device)
    root_logits_tensor, _ = model(root_state)
    root_logits = root_logits_tensor.detach().cpu().numpy().reshape(-1)
    promotion_bias = model.promotion_bias.detach().cpu().numpy().reshape(-1)

    # TT armazena: key -> (depth, flag, value) onde flag: 0=EXACT, 1=LOWER, 2=UPPER
    tt_cache = OrderedDict()
    killer_moves = {}
    best_move_so_far = next(iter(legal))

    class SearchTimeout(Exception):
        pass

    def move_order_score(pos, move, depth=0):
        score = 0.0
        if pos.is_capture(move):
            victim = pos.piece_at(move.to_square)
            attacker = pos.piece_at(move.from_square)
            v_val = PIECE_VALUE.get(victim.piece_type, 1.0) if victim else 1.0
            a_val = PIECE_VALUE.get(attacker.piece_type, 1.0) if attacker else 1.0
            score += 300.0 + v_val * 15.0 - a_val
        elif move == killer_moves.get(depth):
            score += 120.0
        if move.promotion:
            score += 250.0 + PIECE_VALUE.get(move.promotion, 0.0) * 12.0
        return score

    def root_order_score(pos, move, pv_move=None):
        if pv_move is not None and move == pv_move:
            return 1000000.0
        score = move_order_score(pos, move, depth=99)
        if pos.is_castling(move):
            score += 45.0
        score += 1.2 * _move_policy_score(root_logits, promotion_bias, move, root_color)
        return score

    nodes = 0

    def quiesce(pos, alpha, beta, q_depth=6):
        nonlocal nodes
        nodes += 1
        if deadline is not None and (nodes & 1023) == 0 and time.perf_counter() >= deadline:
            raise SearchTimeout()

        stand_pat = _fast_bitboard_eval(pos, pos.turn)
        if q_depth <= 0:
            return stand_pat

        if stand_pat >= beta:
            return beta
        if stand_pat > alpha:
            alpha = stand_pat

        captures = list(pos.generate_legal_captures())
        if not captures:
            return stand_pat

        captures.sort(key=lambda m: move_order_score(pos, m), reverse=True)
        for cap in captures:
            pos.push(cap)
            try:
                val = -quiesce(pos, -beta, -alpha, q_depth - 1)
            finally:
                pos.pop()

            if val >= beta:
                return beta
            if val > alpha:
                alpha = val

        return alpha

    def negamax(pos, depth, alpha, beta, ply):
        nonlocal nodes
        nodes += 1
        if deadline is not None and (nodes & 1023) == 0 and time.perf_counter() >= deadline:
            raise SearchTimeout()

        if pos.is_insufficient_material():
            return 0.0
        if ply > 0 and pos.is_repetition(2):
            return 0.0

        in_check = pos.is_check()
        if in_check:
            depth += 1

        if depth <= 0:
            return quiesce(pos, alpha, beta, q_depth=6)

        alpha_orig = alpha
        key = (pos._transposition_key(), pos.turn)
        cached = tt_cache.get(key)
        if cached is not None:
            c_depth, c_flag, c_val = cached
            if c_depth >= depth:
                tt_cache.move_to_end(key)
                if c_flag == 0:
                    return c_val
                if c_flag == 1 and c_val > alpha:
                    alpha = c_val
                elif c_flag == 2 and c_val < beta:
                    beta = c_val
                if alpha >= beta:
                    return c_val

        moves = list(pos.legal_moves)
        if not moves:
            if in_check:
                return -1000.0 + ply
            return 0.0

        moves.sort(key=lambda m: move_order_score(pos, m, ply), reverse=True)

        # Só limita ramificação em profundidade baixa e NUNCA quando estiver em xeque
        if not in_check and ply >= 2 and depth <= 2 and len(moves) > 22:
            moves = moves[:22]

        best_val = -float("inf")
        for idx, m in enumerate(moves):
            is_quiet = (not pos.is_capture(m)) and (not m.promotion)
            pos.push(m)
            try:
                if idx == 0:
                    val = -negamax(pos, depth - 1, -beta, -alpha, ply + 1)
                else:
                    # Principal Variation Search (PVS) + LMR seguro
                    reduction = 1 if (idx >= 5 and depth >= 3 and is_quiet and not in_check) else 0
                    val = -negamax(pos, depth - 1 - reduction, -alpha - 1e-4, -alpha, ply + 1)
                    if val > alpha and (reduction > 0 or val < beta):
                        val = -negamax(pos, depth - 1, -beta, -alpha, ply + 1)
            finally:
                pos.pop()

            if val > best_val:
                best_val = val
            if val > alpha:
                alpha = val
            if alpha >= beta:
                if is_quiet:
                    killer_moves[ply] = m
                break

        if best_val <= alpha_orig:
            flag = 2  # UPPERBOUND
        elif best_val >= beta:
            flag = 1  # LOWERBOUND
        else:
            flag = 0  # EXACT

        tt_cache[key] = (depth, flag, best_val)
        if len(tt_cache) > 65536:
            tt_cache.popitem(last=False)

        return best_val

    max_target_depth = 9 if time_limit is not None else max(1, int(search_depth))
    start_d = 3 if time_limit is not None else max_target_depth

    for current_depth in range(start_d, max_target_depth + 1):
        try:
            ordered_root = sorted(
                legal,
                key=lambda m: root_order_score(board, m, pv_move=best_move_so_far),
                reverse=True,
            )
            leaf_queue = []
            candidate_records = []
            alpha = -float("inf")
            beta = float("inf")

            for idx, move in enumerate(ordered_root):
                board.push(move)
                try:
                    if board.is_checkmate():
                        return move
                    if board.is_stalemate() or board.is_insufficient_material() or board.is_repetition(2):
                        score = 0.0
                        leaf_state = None
                    else:
                        if idx == 0:
                            score = -negamax(board, current_depth - 1, -beta, -alpha, 1)
                        else:
                            score = -negamax(board, current_depth - 1, -alpha - 1e-4, -alpha, 1)
                            if score > alpha:
                                score = -negamax(board, current_depth - 1, -beta, -alpha, 1)
                        leaf_state = _encode_board_array(board)
                finally:
                    board.pop()

                leaf_idx = None
                if leaf_state is not None and abs(score) < 500.0:
                    leaf_idx = len(leaf_queue)
                    leaf_queue.append((leaf_state, not root_color))

                candidate_records.append((move, score, leaf_idx))
                if score > alpha:
                    alpha = score

            nn_values = None
            if leaf_queue:
                leaf_tensor = torch.from_numpy(np.stack([next(iter(item)) for item in leaf_queue])).to(device)
                nn_values = model(leaf_tensor, value_only=True).detach().cpu().numpy()

            final_scores = []
            for move, exact_val, leaf_idx in candidate_records:
                nn_bonus = 0.0
                if nn_values is not None and leaf_idx is not None:
                    _, leaf_turn = leaf_queue[leaf_idx]
                    v_leaf = float(nn_values[leaf_idx])
                    v_root = v_leaf if leaf_turn == root_color else -v_leaf
                    # Peso sutil (0.08): desempata posições iguais sem nunca entregar peça!
                    nn_bonus = 0.08 * v_root
                prior = _move_policy_score(root_logits, promotion_bias, move, root_color)
                total = exact_val + nn_bonus + 0.02 * float(np.tanh(prior / 8.0))
                final_scores.append(total)

            best_move_so_far = next(iter(candidate_records[int(np.argmax(final_scores))]))

            if deadline is not None and (time.perf_counter() - start_time) > (float(time_limit) * 0.72):
                break

        except SearchTimeout:
            break

    return best_move_so_far


def _random_bot_move(board, rng):
    legal = list(board.legal_moves)
    return rng.choice(legal) if legal else None


def _tactical_bot_move(board, rng):
    best_move = None
    best_score = -float("inf")
    for move in list(board.legal_moves):
        attacker = board.piece_at(move.from_square)
        victim = board.piece_at(move.to_square)
        if board.is_en_passant(move):
            victim = chess.Piece(chess.PAWN, not board.turn)
        capture_value = (
            PIECE_VALUE.get(victim.piece_type, 0)
            if victim else 0
        )
        attacker_value = (
            PIECE_VALUE.get(attacker.piece_type, 0)
            if attacker else 0
        )
        promotion_value = (
            PIECE_VALUE.get(move.promotion, 0)
            if move.promotion else 0
        )
        board.push(move)
        gives_check = board.is_check()
        gives_mate = board.is_checkmate() if gives_check else False
        board.pop()
        score = capture_value * 10.0 - attacker_value * 0.08
        score += promotion_value * 8.0
        score += 3.0 if gives_check else 0.0
        score += 10000.0 if gives_mate else 0.0
        score += rng.random() * 0.001
        if score > best_score:
            best_score = score
            best_move = move
    return best_move


def _expert_static_eval(board, color):
    score = 0.0
    center = (chess.D4, chess.E4, chess.D5, chess.E5)
    for square, piece in board.piece_map().items():
        sign = 1.0 if piece.color == color else -1.0
        score += sign * PIECE_VALUE.get(piece.piece_type, 0)
        if square in center:
            score += sign * 0.22
    for square in center:
        score += 0.06 * (
            len(board.attackers(color, square))
            - len(board.attackers(not color, square))
        )

    if board.is_checkmate():
        return (
            -10000.0
            if board.turn == color
            else 10000.0
        )

    return score


def _expert_bot_move(board, rng):
    color = board.turn
    best_move = None
    best_score = -float("inf")

    for move in list(board.legal_moves):
        board.push(move)

        if board.is_checkmate():
            score = 10000.0
        else:
            replies = list(board.legal_moves)

            if not replies:
                score = _expert_static_eval(board, color)
            else:
                worst_reply = float("inf")

                for reply in replies:
                    board.push(reply)
                    value = _expert_static_eval(board, color)
                    board.pop()

                    if value < worst_reply:
                        worst_reply = value

                score = worst_reply

        board.pop()
        score += rng.random() * 1e-4

        if score > best_score:
            best_score = score
            best_move = move

    return best_move

def _master_static_eval(board, color):
    """Avaliação material, mobilidade, centro e segurança do rei."""
    if board.is_checkmate():
        return (
            -100000.0
            if board.turn == color
            else 100000.0
        )

    if board.is_stalemate():
        return 0.0

    score = 0.0

    for square, piece in board.piece_map().items():
        sign = 1.0 if piece.color == color else -1.0

        score += sign * PIECE_VALUE.get(
            piece.piece_type,
            0,
        )

        if piece.piece_type == chess.KING:
            enemy_king = board.king(not piece.color)

            if enemy_king is not None:
                distance = chess.square_distance(
                    square,
                    enemy_king,
                )

                score += sign * (
                    0.12 * (7 - distance)
                )

    # Mobilidade: quantidade de lances legais para cada lado.
    current_turn = board.turn

    board.turn = color
    own_mobility = board.legal_moves.count()

    board.turn = not color
    enemy_mobility = board.legal_moves.count()

    board.turn = current_turn

    score += 0.025 * (
        own_mobility - enemy_mobility
    )

    # Segurança dos reis: penaliza ataques próximos ao próprio rei.
    own_king = board.king(color)
    enemy_king = board.king(not color)

    if own_king is not None:
        own_attackers = len(
            board.attackers(not color, own_king)
        )

        score -= own_attackers * 0.35

    if enemy_king is not None:
        enemy_attackers = len(
            board.attackers(color, enemy_king)
        )

        score += enemy_attackers * 0.20

    return score


def _master_bot_move(board, rng):
    """Alpha-Beta de profundidade 2 com ordenação tática."""
    color = board.turn

    def ordered_moves(position):
        moves = list(position.legal_moves)

        def move_score(move):
            victim = position.piece_at(move.to_square)
            attacker = position.piece_at(move.from_square)

            capture = (
                PIECE_VALUE.get(victim.piece_type, 0)
                if victim else 0
            )

            attacker_value = (
                PIECE_VALUE.get(attacker.piece_type, 0)
                if attacker else 0
            )

            promotion = (
                PIECE_VALUE.get(move.promotion, 0)
                if move.promotion else 0
            )

            return (
                capture * 10.0
                - attacker_value * 0.05
                + promotion * 8.0
            )

        return sorted(
            moves,
            key=move_score,
            reverse=True,
        )

    def alpha_beta(position, depth, alpha, beta, maximizing):
        if position.is_checkmate():
            return (
                -100000.0
                if position.turn == color
                else 100000.0
            )

        if (
            position.is_stalemate()
            or position.is_insufficient_material()
            or depth <= 0
        ):
            return _master_static_eval(position, color)

        moves = ordered_moves(position)

        if maximizing:
            value = -float("inf")

            for move in moves:
                position.push(move)

                try:
                    child = alpha_beta(
                        position,
                        depth - 1,
                        alpha,
                        beta,
                        False,
                    )
                finally:
                    position.pop()

                value = max(value, child)
                alpha = max(alpha, value)

                if beta <= alpha:
                    break

            return value

        value = float("inf")

        for move in moves:
            position.push(move)

            try:
                child = alpha_beta(
                    position,
                    depth - 1,
                    alpha,
                    beta,
                    True,
                )
            finally:
                position.pop()

            value = min(value, child)
            beta = min(beta, value)

            if beta <= alpha:
                break

        return value

    best_move = None
    best_score = -float("inf")

    for move in ordered_moves(board):
        board.push(move)

        try:
            score = alpha_beta(
                board,
                1,
                -float("inf"),
                float("inf"),
                False,
            )
        finally:
            board.pop()

        score += rng.random() * 0.0001

        if score > best_score:
            best_score = score
            best_move = move

    return best_move

def _strong_bot_eval(board, color):
    """Avaliação completa (material + PST + segurança do rei) para bots de torneio."""
    if board.is_checkmate():
        return -100000.0 if board.turn == color else 100000.0
    if board.is_stalemate() or board.is_insufficient_material():
        return 0.0

    score = 0.0
    for sq, piece in board.piece_map().items():
        val = PIECE_VALUE.get(piece.piece_type, 0) * 100.0
        pst_idx = (sq ^ 56) if piece.color == chess.WHITE else sq
        pst = _PST_TABLES[piece.piece_type][pst_idx]
        if piece.color == color:
            score += val + pst
        else:
            score -= val + pst

    own_king = board.king(color)
    enemy_king = board.king(not color)
    if own_king is not None:
        score -= len(board.attackers(not color, own_king)) * 28.0
    if enemy_king is not None:
        score += len(board.attackers(color, enemy_king)) * 28.0

    return score


def _quiescence_bot(board, color, alpha, beta, q_depth=2):
    stand_pat = _strong_bot_eval(board, color)
    maximizing = (board.turn == color)
    if q_depth <= 0:
        return stand_pat

    if maximizing:
        if stand_pat >= beta:
            return beta
        if stand_pat > alpha:
            alpha = stand_pat
    else:
        if stand_pat <= alpha:
            return alpha
        if stand_pat < beta:
            beta = stand_pat

    captures = [m for m in board.legal_moves if board.is_capture(m)]
    if not captures:
        return stand_pat

    captures.sort(
        key=lambda m: (
            PIECE_VALUE.get(board.piece_at(m.to_square).piece_type, 1)
            if board.piece_at(m.to_square) else 1
        ),
        reverse=True,
    )

    for move in captures[:6]:
        board.push(move)
        try:
            score = _quiescence_bot(board, color, alpha, beta, q_depth - 1)
        finally:
            board.pop()

        if maximizing:
            if score > alpha:
                alpha = score
            if alpha >= beta:
                return beta
        else:
            if score < beta:
                beta = score
            if beta <= alpha:
                return alpha

    return alpha if maximizing else beta


def _search_reference_bot(board, rng, depth=2, q_depth=2):
    color = board.turn
    tt_cache = OrderedDict()

    def order_key(pos, m):
        s = 0
        if pos.is_capture(m):
            vic = pos.piece_at(m.to_square)
            att = pos.piece_at(m.from_square)
            v_val = PIECE_VALUE.get(vic.piece_type, 1) if vic else 1
            a_val = PIECE_VALUE.get(att.piece_type, 1) if att else 1
            s += 1000 + v_val * 100 - a_val * 10
        if m.promotion:
            s += 800 + PIECE_VALUE.get(m.promotion, 0) * 80
        piece = pos.piece_at(m.from_square)
        if piece is not None:
            pst_idx = (m.to_square ^ 56) if piece.color == chess.WHITE else m.to_square
            s += _PST_TABLES[piece.piece_type][pst_idx]
        return s

    def alphabeta(pos, d, alpha, beta):
        if pos.is_checkmate():
            return -100000.0 - d * 100.0 if pos.turn == color else 100000.0 + d * 100.0
        if pos.is_stalemate() or pos.is_insufficient_material() or pos.is_repetition(2):
            return 0.0
        if d <= 0:
            return _quiescence_bot(pos, color, alpha, beta, q_depth)

        key = (pos._transposition_key(), d, pos.turn == color)
        if key in tt_cache:
            tt_cache.move_to_end(key)
            return tt_cache[key]

        moves = sorted(pos.legal_moves, key=lambda m: order_key(pos, m), reverse=True)
        if not moves:
            return 0.0

        if pos.turn == color:
            val = -float("inf")
            for m in moves:
                pos.push(m)
                try:
                    val = max(val, alphabeta(pos, d - 1, alpha, beta))
                finally:
                    pos.pop()
                alpha = max(alpha, val)
                if beta <= alpha:
                    break
        else:
            val = float("inf")
            for m in moves:
                pos.push(m)
                try:
                    val = min(val, alphabeta(pos, d - 1, alpha, beta))
                finally:
                    pos.pop()
                beta = min(beta, val)
                if beta <= alpha:
                    break

        tt_cache[key] = val
        if len(tt_cache) > 4096:
            tt_cache.popitem(last=False)
        return val

    root_moves = sorted(board.legal_moves, key=lambda m: order_key(board, m), reverse=True)
    if not root_moves:
        return None

    best_move = root_moves[0]
    alpha = -float("inf")
    beta = float("inf")

    for move in root_moves:
        board.push(move)
        try:
            score = alphabeta(board, depth - 1, alpha, beta)
        finally:
            board.pop()

        score += rng.random() * 0.4
        if score > alpha:
            alpha = score
            best_move = move

    return best_move


def _club_player_bot_move(board, rng):
    """Jogador de Clube (1650 Elo FIDE): Alpha-Beta prof. 2 + Quiescence."""
    return _search_reference_bot(board, rng, depth=2, q_depth=2)


def _candidate_master_bot_move(board, rng):
    """Candidato a Mestre (1950 Elo FIDE): Alpha-Beta prof. 3 + Quiescence."""
    return _search_reference_bot(board, rng, depth=3, q_depth=2)


def _fide_master_bot_move(board, rng):
    """Mestre FIDE (2300 Elo FIDE): Alpha-Beta prof. 3 + Quiescence profunda (4)."""
    return _search_reference_bot(board, rng, depth=3, q_depth=4)


def _international_master_bot_move(board, rng):
    """Mestre Internacional (2450 Elo FIDE): Alpha-Beta prof. 4 + Quiescence profunda (4)."""
    return _search_reference_bot(board, rng, depth=4, q_depth=4)


def _play_reference_game(
    model,
    bot_kind,
    model_color,
    rng,
    move_delay=0.0,
    training_samples=None,
):
    board = chess.Board()
    game_samples = []
    distill_bots = (
        "casual",
        "intermediate",
        "tournament",
        "club_player",
        "candidate_master",
        "fide_master",
        "international_master",
    )

    for _ in range(BENCHMARK_MAX_PLIES):
        if board.is_game_over(claim_draw=True):
            break

        if board.turn == model_color:
            state_before = (
                _encode_board_array(board)
                if bot_kind in distill_bots
                and training_samples is not None
                else None
            )

            mover = board.turn
            move = _model_move(model, board)

            if move is not None and state_before is not None:
                game_samples.append((
                    state_before,
                    move_to_action(move, mover),
                    mover,
                ))

        elif bot_kind == "novice":
            move = _random_bot_move(board, rng)

        elif bot_kind == "intermediate":
            move = _expert_bot_move(board, rng)

        elif bot_kind == "tournament":
            move = _master_bot_move(board, rng)

        elif bot_kind == "club_player":
            move = _club_player_bot_move(board, rng)

        elif bot_kind == "candidate_master":
            move = _candidate_master_bot_move(board, rng)

        elif bot_kind == "fide_master":
            move = _fide_master_bot_move(board, rng)

        elif bot_kind == "international_master":
            move = _international_master_bot_move(board, rng)

        else:
            move = _tactical_bot_move(board, rng)

        if move is None:
            break

        board.push(move)

        if move_delay > 0:
            time.sleep(float(move_delay))

    result = (
        board.result(claim_draw=True)
        if board.is_game_over(claim_draw=True)
        else "1/2-1/2"
    )

    if result == "1/2-1/2":
        outcome = "draw"
        winner = None
    else:
        winner = (
            chess.WHITE
            if result == "1-0"
            else chess.BLACK
        )

        model_won = winner == model_color
        outcome = "win" if model_won else "loss"

    if (
        training_samples is not None
        and bot_kind in distill_bots
        and outcome in ("win", "draw")
    ):
        for state, action, mover in game_samples:
            target = (
                0.0
                if winner is None
                else 1.0 if mover == winner else -1.0
            )

            training_samples.append((
                state,
                int(action),
                float(target),
                1.5,
                1.0,
            ))

    return outcome


def run_reference_benchmark(
    model,
    move_delay=0.0,
    training_samples=None,
):
    model.eval()
    rng = random.Random()
    results = {}

    for bot_kind, opponent_rating in REFERENCE_BOTS.items():
        record = {
            "rating": opponent_rating,
            "wins": 0,
            "draws": 0,
            "losses": 0,
        }

        for game_index in range(BENCHMARK_GAMES_PER_BOT):
            model_color = game_index % 2 == 0

            outcome = _play_reference_game(
                model,
                bot_kind,
                model_color,
                rng,
                move_delay=move_delay,
                training_samples=training_samples,
            )

            record[
                {"win": "wins", "draw": "draws", "loss": "losses"}[outcome]
            ] += 1

        results[bot_kind] = record

    return results


class ChessBrain:
    def __init__(self):
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        if self.device.type == "cuda":
            torch.backends.cudnn.benchmark = True

        self.model = PolicyValueNet().to(self.device)

        self.population = []
        self.fitness = []

        self.games_trained = 0
        self.rating = 400.0
        self.elo_history = []
        self.generation = 0
        self.best_index = 0

        self.result_totals = {
            "white": 0,
            "draw": 0,
            "black": 0,
        }

        self.recent_results = []
        self.pending_score_totals = [0.0] * POPULATION_SIZE
        self.pending_game_counts = [0] * POPULATION_SIZE
        self.next_genome_index = 0

        self.last_benchmark_generation = 0
        self.benchmark_history = []

        self.training_samples = deque(maxlen=2048)
        self._training_sample_seen = 0

        self.last_loss = 0.0
        self._last_save_time = time.monotonic()

        self._save_lock = threading.Lock()
        self._model_lock = threading.RLock()

        self._legacy_checkpoint = None
        self._saved_optimizer_states = []

        self.bongcloud_reset_done = True

        loaded = self.load_checkpoint()

        if not loaded:
            self.population = [
                PolicyValueNet().to(self.device)
                for _ in range(POPULATION_SIZE)
            ]

            for net in self.population:
                net.load_state_dict(self.model.state_dict())

            for index in range(1, POPULATION_SIZE):
                self.mutate(
                    self.population[index],
                    scale=0.01,
                )

            self.fitness = [0.0] * POPULATION_SIZE

        if (
            loaded
            and not self.bongcloud_reset_done
            and self.last_loss < 0.05
        ):
            print(
                "[RECUPERAÇÃO] Policy collapse detectado "
                f"(loss={self.last_loss:.6f}); "
                "diversificando 19 indivíduos."
            )

            for index in range(1, len(self.population)):
                self.mutate(
                    self.population[index],
                    scale=0.015,
                )

            self.training_samples.clear()
            self._training_sample_seen = 0
            self._saved_optimizer_states = []
            self.last_loss = 0.05

            self.model.load_state_dict(
                self.population[0].state_dict()
            )

        self.optimizers = [
            torch.optim.Adam(
                net.parameters(),
                lr=3e-5,
                weight_decay=1e-4,
            )
            for net in self.population
        ]

        for index, saved in enumerate(
            getattr(self, "_saved_optimizer_states", [])
        ):
            if index < len(self.optimizers) and saved:
                try:
                    self.optimizers[index].load_state_dict(saved)

                    for group in self.optimizers[index].param_groups:
                        group["lr"] = 3e-5
                        group["weight_decay"] = 1e-4

                except Exception as exc:
                    print(
                        f"[AVISO] Estado Adam do indivíduo "
                        f"{index} reiniciado: {exc}"
                    )

        self._saved_optimizer_states = []

        global VALUE_HEAD_WEIGHT
        VALUE_HEAD_WEIGHT = (
            0.55
            if 0 < self.last_loss < 5.0
            else 0.35
        )

        self._refresh_batched_state()

    def _refresh_batched_state(self):
        with self._model_lock:
            for net in self.population:
                net.eval()

            params, buffers = torch.func.stack_module_state(
                self.population
            )

            self._batched_params = params
            self._batched_buffers = buffers

    @property
    def device_name(self):
        if self.device.type == "cuda":
            return f"CUDA: {torch.cuda.get_device_name(0)}"

        return "CPU"

    @torch.inference_mode()
    def batch_choose(self, boards, individual_ids, temperature=0.14):
        count = len(boards)

        if not count:
            return [], []

        states = np.stack([
            _encode_board_array(board)
            for board in boards
        ])

        legal_moves = [
            [
                (move.from_square, move.to_square, move.promotion or 0)
                for move in board.legal_moves
            ]
            for board in boards
        ]

        fens = [board.fen() for board in boards]
        plies = [_board_ply(board) for board in boards]
        turns = [board.turn for board in boards]

        return self.batch_choose_encoded(
            states,
            individual_ids,
            legal_moves,
            temperature,
            fens=fens,
            plies=plies,
            turns=turns,
        )

    def _forward_population_locked(
        self,
        states,
        individual_ids,
        value_only=False,
    ):
        if len(states) == 0:
            return (
                np.empty((0,), dtype=np.float32),
                np.empty((0,), dtype=np.float32),
                np.empty((0, 5), dtype=np.float32),
            )

        cpu_states = torch.from_numpy(
            np.ascontiguousarray(states, dtype=np.float32)
        )

        state_tensor = cpu_states.to(self.device)

        genome_ids = torch.as_tensor(
            individual_ids,
            device=self.device,
            dtype=torch.long,
        )

        if (
            genome_ids.numel() != len(states)
            or torch.any(genome_ids < 0)
            or torch.any(genome_ids >= POPULATION_SIZE)
        ):
            raise ValueError(
                "individual_ids contém índice fora da população"
            )

        params = {
            name: value.index_select(0, genome_ids)
            for name, value in self._batched_params.items()
        }

        buffers = {
            name: value.index_select(0, genome_ids)
            for name, value in self._batched_buffers.items()
        }

        base_model = self.population[0]

        if value_only:
            def forward_one(sample_params, sample_buffers, state):
                return torch.func.functional_call(
                    base_model,
                    (sample_params, sample_buffers),
                    (state.unsqueeze(0),),
                    {"value_only": True},
                ).squeeze(0)

            values = torch.vmap(
                forward_one,
                in_dims=(0, 0, 0),
            )(params, buffers, state_tensor)

            return (
                None,
                values.detach().cpu().numpy(),
                params["promotion_bias"].detach().cpu().numpy(),
            )

        def forward_one(sample_params, sample_buffers, state):
            logits, value = torch.func.functional_call(
                base_model,
                (sample_params, sample_buffers),
                (state.unsqueeze(0),),
            )

            return logits.squeeze(0), value.squeeze(0)

        logits, values = torch.vmap(
            forward_one,
            in_dims=(0, 0, 0),
        )(params, buffers, state_tensor)

        return (
            logits.detach().cpu().numpy(),
            values.detach().cpu().numpy(),
            params["promotion_bias"].detach().cpu().numpy(),
        )

    @torch.inference_mode()
    def batch_choose_encoded(
        self,
        states,
        individual_ids,
        legal_moves,
        temperature=0.14,
        fens=None,
        batch_size=None,
        tactical_bonuses=None,
        turbo=None,
        plies=None,
        turns=None,
    ):
        count = len(individual_ids)

        if count == 0:
            return [], []

        if len(states) != count or len(legal_moves) != count:
            raise ValueError(
                "states, individual_ids e legal_moves "
                "precisam ter o mesmo tamanho"
            )

        if fens is not None and len(fens) != count:
            raise ValueError(
                "fens precisa ter o mesmo tamanho de states"
            )

        if plies is not None and len(plies) != count:
            raise ValueError(
                "plies precisa ter o mesmo tamanho de states"
            )

        if turns is not None and len(turns) != count:
            raise ValueError(
                "turns precisa ter o mesmo tamanho de states"
            )

        if tactical_bonuses is not None and len(tactical_bonuses) != count:
            raise ValueError(
                "tactical_bonuses precisa ter o mesmo tamanho de states"
            )

        with self._model_lock:
            root_logits, root_values, root_promos = (
                self._forward_population_locked(
                    states,
                    individual_ids,
                )
            )

            turbo_policy_only = (
                                float(temperature) >= 0.40
                if turbo is None
                else bool(turbo)
            )

            lookahead_rows = []
            chosen = [None] * count

            for row in range(count):
                moves = legal_moves[row]

                if not moves:
                    continue

                row_bonuses = (
                    tactical_bonuses[row]
                    if tactical_bonuses is not None
                    else None
                )

                if turns is not None:
                    root_turn = bool(turns[row])
                elif fens is not None and fens[row] is not None:
                    root_turn = (
                        chess.WHITE
                        if fens[row].split(" ", 2)[1] == "w"
                        else chess.BLACK
                    )
                else:
                    root_turn = chess.WHITE

                has_ply_info = (
                    plies is not None
                    or (
                        fens is not None
                        and fens[row] is not None
                    )
                )

                if plies is not None:
                    current_ply = int(plies[row])
                elif fens is not None and fens[row] is not None:
                    current_ply = _fen_ply(fens[row])
                else:
                    current_ply = 0

                opening_limit = 12 if turbo_policy_only else 4

                if has_ply_info and current_ply < 6:
                    chosen[row] = random.choice(moves)

                elif has_ply_info and current_ply < opening_limit:
                    chosen[row] = _choose_top_k_policy_move(
                        moves,
                        root_logits[row],
                        root_promos[row],
                        top_k=4,
                        temperature=1.2,
                        tactical_bonuses=row_bonuses,
                        turn=root_turn,
                    )

                elif turbo_policy_only:
                    chosen[row] = _choose_policy_move(
                        moves,
                        root_logits[row],
                        root_promos[row],
                        temperature,
                        tactical_bonuses=row_bonuses,
                        turn=root_turn,
                        dirichlet_noise=True,
                    )

                elif fens is not None and fens[row] is not None:
                    lookahead_rows.append(row)

                else:
                    chosen[row] = _choose_policy_move(
                        moves,
                        root_logits[row],
                        root_promos[row],
                        temperature,
                        tactical_bonuses=row_bonuses,
                        turn=root_turn,
                    )

            if not lookahead_rows:
                return (
                    chosen,
                    root_values.astype(float).tolist(),
                )

            records = []
            opponent_states = []
            opponent_genomes = []
            opponent_owners = []

            for row in lookahead_rows:
                move_tuples = legal_moves[row]
                board = chess.Board(fens[row])

                moves = [
                    chess.Move(
                        int(frm),
                        int(to),
                        promotion=int(promo) if promo else None,
                    )
                    for frm, to, promo in move_tuples
                ]

                selected = _candidate_moves(
                    board,
                    moves,
                    root_logits[row],
                    root_promos[row],
                    top_n=5,
                )

                genome = int(individual_ids[row])

                for move in selected:
                    after = board.copy(stack=False)
                    after.push(move)

                    terminal = _terminal_score(
                        after,
                        board.turn,
                    )

                    responses = (
                        []
                        if terminal is not None
                        else list(after.legal_moves)
                    )

                    record = {
                        "row": row,
                        "genome": genome,
                        "move": (
                            move.from_square,
                            move.to_square,
                            move.promotion or 0,
                        ),
                        "root_color": board.turn,
                        "prior": _move_policy_score(
                            root_logits[row],
                            root_promos[row],
                            move,
                            board.turn,
                        ),
                        "terminal": terminal,
                        "after": after,
                        "responses": responses,
                        "leaves": [],
                    }

                    record_index = len(records)
                    records.append(record)

                    if terminal is None:
                        if responses:
                            opponent_owners.append(record_index)
                            opponent_states.append(
                                _encode_board_array(after)
                            )
                            opponent_genomes.append(genome)
                        else:
                            record["terminal"] = 0.0

            leaf_states = []
            leaf_genomes = []
            leaf_materials = []
            leaf_turns = []

            if opponent_states:
                opp_logits, _, opp_promos = (
                    self._forward_population_locked(
                        np.stack(opponent_states),
                        opponent_genomes,
                    )
                )

                for opp_row, record_index in enumerate(opponent_owners):
                    record = records[record_index]

                    best_replies = sorted(
                        record["responses"],
                        key=lambda move: _move_policy_score(
                            opp_logits[opp_row],
                            opp_promos[opp_row],
                            move,
                            record["after"].turn,
                        ),
                        reverse=True,
                    )[:3]

                    after = record["after"]

                    for reply in best_replies:
                        after.push(reply)

                        try:
                            terminal = _terminal_score(
                                after,
                                record["root_color"],
                            )

                            if terminal is not None:
                                record["leaves"].append(
                                    (None, terminal, 0.0, after.turn)
                                )
                            else:
                                material = _material_score(
                                    after,
                                    record["root_color"],
                                )

                                leaf_index = len(leaf_states)

                                record["leaves"].append((
                                    leaf_index,
                                    None,
                                    material,
                                    after.turn,
                                ))

                                leaf_states.append(
                                    _encode_board_array(after)
                                )
                                leaf_genomes.append(record["genome"])
                                leaf_materials.append(material)
                                leaf_turns.append(after.turn)

                        finally:
                            after.pop()

            leaf_values = np.empty(
                (0,),
                dtype=np.float32,
            )

            if leaf_states:
                _, leaf_values, _ = self._forward_population_locked(
                    np.stack(leaf_states),
                    leaf_genomes,
                    value_only=True,
                )

            scores_by_row = {
                row: [] for row in lookahead_rows
            }

            moves_by_row = {
                row: [] for row in lookahead_rows
            }

            for record in records:
                outcomes = []

                if record["terminal"] is not None:
                    outcomes.append(float(record["terminal"]))

                for leaf_index, terminal, material, leaf_turn in record["leaves"]:
                    if terminal is not None:
                        outcomes.append(float(terminal))
                    else:
                        outcomes.append(_leaf_score(
                            float(leaf_values[leaf_index]),
                            material,
                            record["root_color"],
                            leaf_turn,
                        ))

                worst_reply = min(outcomes) if outcomes else 0.0

                score = (
                    worst_reply
                    + 0.02 * float(np.tanh(record["prior"] / 8.0))
                )

                scores_by_row[record["row"]].append(score)
                moves_by_row[record["row"]].append(record["move"])

            for row in lookahead_rows:
                options = moves_by_row[row]

                if not options:
                    continue

                scores = np.asarray(
                    scores_by_row[row],
                    dtype=np.float64,
                )

                if temperature <= 0:
                    selected_index = int(np.argmax(scores))
                else:
                    scaled = (
                        scores - scores.max()
                    ) / max(0.02, float(temperature))

                    probabilities = np.exp(
                        np.clip(scaled, -60.0, 0.0)
                    )
                    probabilities /= probabilities.sum()

                    selected_index = int(
                        np.random.choice(len(options), p=probabilities)
                    )

                chosen[row] = options[selected_index]

            return (
                chosen,
                root_values.astype(float).tolist(),
            )

    @staticmethod
    @torch.no_grad()
    def mutate(model, scale=0.005):
        for param in model.parameters():
            param.add_(torch.randn_like(param) * scale)

    def record_completed_matches(self, matches):
        """Registra Policy e Value com pesos independentes."""
        accepted = []

        for match in matches:
            result = match.get("status")
            genome = int(match.get("genome", -1))

            if (
                result not in ("1-0", "1/2-1/2", "0-1")
                or not 0 <= genome < POPULATION_SIZE
            ):
                continue

            accepted.append((
                result,
                genome,
                float(match.get("fitness", 0.0)),
                match.get("samples", []),
            ))

        if not accepted:
            return

        with self._save_lock:
            for result, genome, fitness, samples in accepted:
                self.pending_score_totals[genome] += fitness
                self.pending_game_counts[genome] += 1
                self.games_trained += 1

                if result == "1-0":
                    self.result_totals["white"] += 1
                elif result == "0-1":
                    self.result_totals["black"] += 1
                else:
                    self.result_totals["draw"] += 1

                self.recent_results.append(result)

                winner = (
                    chess.WHITE if result == "1-0"
                    else chess.BLACK if result == "0-1"
                    else None
                )

                for sample in samples:
                    state, action, mover, weight = sample[:4]
                    sample_type = (
                        sample[5]
                        if len(sample) >= 6
                        else None
                    )

                    if sample_type == "material":
                        target = float(sample[4])
                        policy_weight = 0.0
                        value_weight = float(weight)

                    elif sample_type == "mate":
                        target = (
                            0.0
                            if winner is None
                            else 1.0 if bool(mover) == bool(winner)
                            else -1.0
                        )

                        policy_weight = (
                            float(weight)
                            if winner is not None and bool(mover) == bool(winner)
                            else 0.0
                        )

                        value_weight = 1.0

                    else:
                        target = (
                            0.0
                            if winner is None
                            else 1.0 if bool(mover) == bool(winner)
                            else -1.0
                        )

                        policy_weight = (
                            float(weight)
                            if winner is not None and bool(mover) == bool(winner)
                            else 0.0
                        )

                        value_weight = 1.0

                    self._training_sample_seen += 1

                    self.training_samples.append((
                        np.asarray(state, dtype=np.float32),
                        int(action),
                        float(target),
                        float(policy_weight),
                        float(value_weight),
                    ))

            self.recent_results = self.recent_results[-100:]

    def record_benchmark_samples(self, samples):
        if not samples:
            return

        with self._save_lock:
            for sample in samples:
                state, action, target, policy_weight = sample[:4]

                value_weight = (
                    float(sample[4])
                    if len(sample) >= 5
                    else 1.0
                )

                self.training_samples.append((
                    np.asarray(state, dtype=np.float32),
                    int(action),
                    float(target),
                    float(policy_weight),
                    float(value_weight),
                ))

                self._training_sample_seen += 1

    def training_cycle_ready(self):
        return (
            len(self.pending_game_counts) == POPULATION_SIZE
            and all(count >= 1 for count in self.pending_game_counts)
        )

    def _train_models_locked(
        self,
        indices,
        max_samples=2048,
        batch_size=128,
        epochs=1,
    ):
        global VALUE_HEAD_WEIGHT

        samples = list(self.training_samples)[
            -max(1, int(max_samples)):
        ]

        # 64 posições sintéticas para melhorar a conversão
        # de finais básicos ganhos.
        samples.extend(
            _generate_endgame_drills(64)
        )

        if not samples:
            self.last_loss = 0.0
            VALUE_HEAD_WEIGHT = 0.35
            return self.last_loss

        states = np.stack([
            item[0] for item in samples
        ]).astype(np.float32, copy=False)

        actions = np.asarray(
            [item[1] for item in samples],
            dtype=np.int64,
        )

        targets = np.asarray(
            [item[2] for item in samples],
            dtype=np.float32,
        )

        policy_weights = np.asarray([
            item[3] if len(item) >= 5 else item[3]
            for item in samples
        ], dtype=np.float32)

        value_weights = np.asarray([
            item[4] if len(item) >= 5 else 1.0
            for item in samples
        ], dtype=np.float32)
        x = torch.from_numpy(states).to(self.device)
        y_action = torch.from_numpy(actions).to(self.device)
        y_value = torch.from_numpy(targets).to(self.device)

        policy_sample_weights = torch.from_numpy(
            policy_weights
        ).to(self.device)

        value_sample_weights = torch.from_numpy(
            value_weights
        ).to(self.device)

        losses = []

        for model_index in indices:
            model = self.population[model_index]
            optimizer = self.optimizers[model_index]

            model.train()

            for _epoch in range(max(1, int(epochs))):
                order = torch.randperm(
                    len(samples),
                    device=self.device,
                )

                for start in range(
                    0,
                    len(samples),
                    max(1, int(batch_size)),
                ):
                    batch_indices = order[
                        start:start + max(1, int(batch_size))
                    ]

                    logits, values = model(
                        x.index_select(0, batch_indices)
                    )

                    batch_value_weights = (
                        value_sample_weights.index_select(
                            0,
                            batch_indices,
                        )
                    )

                    value_each = F.mse_loss(
                        values,
                        y_value.index_select(0, batch_indices),
                        reduction="none",
                    )

                    value_loss = (
                        value_each * batch_value_weights
                    ).sum() / batch_value_weights.sum().clamp_min(1e-6)

                    batch_policy_weights = (
                        policy_sample_weights.index_select(
                            0,
                            batch_indices,
                        )
                    )

                    policy_each = F.cross_entropy(
                        logits,
                        y_action.index_select(0, batch_indices),
                        reduction="none",
                        label_smoothing=0.1,
                    )

                    policy_loss = (
                        policy_each * batch_policy_weights
                    ).sum() / batch_policy_weights.sum().clamp_min(1e-6)

                    probabilities = torch.softmax(
                        logits,
                        dim=-1,
                    )

                    entropy = -(
                        probabilities
                        * torch.log(probabilities.clamp_min(1e-8))
                    ).sum(dim=-1)

                    normalized_entropy = (
                        entropy / np.log(max(2, logits.shape[-1]))
                    )

                    if batch_policy_weights.sum() > 0:
                        entropy_bonus = (
                            normalized_entropy
                            * (batch_policy_weights > 0).float()
                        ).sum() / (
                            (batch_policy_weights > 0).float()
                            .sum().clamp_min(1.0)
                        )
                    else:
                        entropy_bonus = torch.zeros(
                            (),
                            device=self.device,
                        )

                    loss = (
                        value_loss
                        + policy_loss
                        - 0.15 * entropy_bonus
                    )

                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()

                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(),
                        1.0,
                    )

                    optimizer.step()
                    losses.append(float(loss.detach().cpu()))

            model.eval()

        self.last_loss = (
            float(np.mean(losses))
            if losses
            else 0.0
        )

        VALUE_HEAD_WEIGHT = (
            0.55 if self.last_loss < 5.0 else 0.35
        )

        return self.last_loss

    def evolve(self, scores):
        scores = [float(score) for score in scores]

        if len(scores) != POPULATION_SIZE:
            raise ValueError(
                f"Esperados {POPULATION_SIZE} scores, "
                f"recebidos {len(scores)}"
            )

        with self._save_lock, self._model_lock:
            if not self.training_cycle_ready():
                raise RuntimeError(
                    "Evolução bloqueada: todos os 20 indivíduos "
                    "precisam concluir pelo menos uma partida."
                )

            self.fitness = scores

            ranked = sorted(
                range(POPULATION_SIZE),
                key=lambda index: scores[index],
                reverse=True,
            )

            gradient_loss = self._train_models_locked(
                ranked[:4],
                max_samples=2048,
                epochs=2,
            )

            next_population = [
                self.population[index]
                for index in ranked[:4]
            ]

            next_optimizers = [
                self.optimizers[index]
                for index in ranked[:4]
            ]

            while len(next_population) < POPULATION_SIZE:
                ia, ib = random.sample(ranked[:10], 2)

                parent_a = self.population[ia].state_dict()
                parent_b = self.population[ib].state_dict()

                child = PolicyValueNet().to(self.device)
                child_state = {}

                with torch.no_grad():
                    for key, a in parent_a.items():
                        b = parent_b[key]

                        if a.is_floating_point():
                            mask = (
                                torch.rand(a.shape, device=a.device) < 0.5
                            )

                            value = torch.where(
                                mask,
                                a,
                                b,
                            ).clone()

                            if key != "promotion_bias":
                                value.add_(
                                    torch.randn_like(value) * 0.0008
                                )

                            child_state[key] = value
                        else:
                            child_state[key] = a.clone()

                child.load_state_dict(child_state)
                next_population.append(child)

                next_optimizers.append(
                    torch.optim.Adam(
                        child.parameters(),
                        lr=3e-5,
                        weight_decay=1e-4,
                    )
                )

            self.population = next_population
            self.optimizers = next_optimizers

            self._refresh_batched_state()

            self.best_index = 0
            self.model.load_state_dict(
                self.population[0].state_dict()
            )

            self.generation += 1

            self.pending_score_totals = [0.0] * POPULATION_SIZE
            self.pending_game_counts = [0] * POPULATION_SIZE


            self.elo_history.append((
                float(self.games_trained),
                float(self.rating),
            ))

            self.elo_history = self.elo_history[-500:]

            print(
                f"[GRADIENTE] {len(self.training_samples)} posições | "
                f"4 elites | 1 época | Adam persistente "
                f"lr=3e-5, wd=1e-4 | loss={gradient_loss:.5f}"
            )

    def train_from_selfplay(
        self,
        max_samples=2048,
        elites=4,
        batch_size=128,
        epochs=1,
    ):
        with self._save_lock, self._model_lock:
            loss = self._train_models_locked(
                list(range(
                    min(
                        max(1, int(elites)),
                        len(self.population),
                    )
                )),
                max_samples=max_samples,
                batch_size=batch_size,
                epochs=epochs,
            )

            self.model.load_state_dict(
                self.population[0].state_dict()
            )

            self._refresh_batched_state()
            return loss

    def make_benchmark_model(self):
        with self._model_lock:
            source = self.population[self.best_index]
            model = PolicyValueNet().to(self.device)

            state = {
                key: value.detach().to(
                    device=self.device
                ).clone()
                for key, value in source.state_dict().items()
            }

            model.load_state_dict(state)
            model.eval()

            return model

    def apply_benchmark_results(self, results, generation):
        delta = 0.0
        details = []

        for bot_kind, opponent_rating in REFERENCE_BOTS.items():
            record = results.get(bot_kind, {})

            wins = int(record.get("wins", 0))
            draws = int(record.get("draws", 0))
            losses = int(record.get("losses", 0))

            games = wins + draws + losses

            if games <= 0:
                continue

            actual = wins + 0.5 * draws

            expected_one = 1.0 / (
                1.0
                + 10.0 ** (
                    (opponent_rating - self.rating) / 400.0
                )
            )

            delta += ELO_K * (
                actual - games * expected_one
            )

            details.append((
                bot_kind,
                wins,
                draws,
                losses,
            ))

        with self._save_lock:
            old_rating = self.rating
            self.rating += delta

            self.last_benchmark_generation = max(
                self.last_benchmark_generation,
                int(generation),
            )


            self.elo_history.append((
                float(self.games_trained),
                float(self.rating),
            ))

            self.elo_history = self.elo_history[-500:]

            self.benchmark_history.append({
                "generation": int(generation),
                "delta": float(delta),
                "rating": float(self.rating),
                "results": {
                    key: {
                        k: int(v)
                        for k, v in value.items()
                    }
                    for key, value in results.items()
                },
            })

            self.benchmark_history = self.benchmark_history[-100:]

        print(
            f"[ELO REAL G{generation}] "
            f"{old_rating:.1f} -> {self.rating:.1f} "
            f"({delta:+.1f}) | "
            + " | ".join(
                f"{name}: {wins}V {draws}E {losses}D"
                for name, wins, draws, losses in details
            )
        )

        return delta

    def _cpu_fp16_state(self, model):
        with self._model_lock:
            result = {}

            for key, tensor in model.state_dict().items():
                cpu = tensor.detach().to(device="cpu")

                result[key] = (
                    cpu.to(dtype=torch.float16)
                    if cpu.is_floating_point()
                    else cpu.clone()
                )

            return result

    @staticmethod
    def _optimizer_state_to_cpu(state):
        result = {}

        for key, value in state.items():
            if key == "state":
                result[key] = {
                    parameter_id: {
                        name: (
                            tensor.detach().cpu().clone()
                            if torch.is_tensor(tensor)
                            else tensor
                        )
                        for name, tensor in parameter_state.items()
                    }
                    for parameter_id, parameter_state in value.items()
                }
            else:
                result[key] = value

        return result

    def should_auto_save(self, seconds=300):
        return (
            time.monotonic() - self._last_save_time
            >= seconds
        )

    def _save_checkpoint_unlocked(self, path=CHECKPOINT_PATH):
        path = Path(path)

        tmp = (
            path.with_name("checkpoint.tmp")
            if path.name == "checkpoint.pth"
            else path.with_name(path.name + ".tmp")
        )

        path.parent.mkdir(parents=True, exist_ok=True)

        with self._model_lock:
            population_snapshot = list(self.population)

            model_state = self._cpu_fp16_state(
                population_snapshot[0]
            )

            population_states = [
                self._cpu_fp16_state(net)
                for net in population_snapshot
            ]

        packed_samples = []

        for sample in list(self.training_samples)[-2048:]:
            state, action, target = sample[:3]

            policy_weight = (
                float(sample[3])
                if len(sample) >= 4
                else 1.0
            )

            value_weight = (
                float(sample[4])
                if len(sample) >= 5
                else 1.0
            )

            packed_samples.append((
                np.packbits(
                    np.asarray(state, dtype=np.uint8).reshape(-1)
                ).tobytes(),
                int(action),
                float(target),
                policy_weight,
                value_weight,
            ))

        payload = {
            "format": CHECKPOINT_FORMAT,
            "model": model_state,
            "population": population_states,

            "games_trained": int(self.games_trained),
            "generation": int(self.generation),
            "rating": float(self.rating),
            "rating_method": RATING_METHOD,

            "elo_history": [
                (float(a), float(b))
                for a, b in self.elo_history[-500:]
            ],

            "fitness": [
                float(x)
                for x in self.fitness[:POPULATION_SIZE]
            ],

            "result_totals": {
                key: int(value)
                for key, value in self.result_totals.items()
            },

            "recent_results": list(self.recent_results[-100:]),

            "pending_score_totals": [
                float(x)
                for x in self.pending_score_totals
            ],

            "pending_game_counts": [
                int(x)
                for x in self.pending_game_counts
            ],

            "next_genome_index": int(self.next_genome_index),
            "last_benchmark_generation": int(
                self.last_benchmark_generation
            ),

            "benchmark_history": list(
                self.benchmark_history[-100:]
            ),

            "training_samples": packed_samples,

            "optimizer_states": [
                self._optimizer_state_to_cpu(opt.state_dict())
                for opt in self.optimizers
            ],

            "last_loss": float(self.last_loss),
            "training_sample_seen": int(
                self._training_sample_seen
            ),

            "bongcloud_reset_done": bool(
                self.bongcloud_reset_done
            ),
            "clean_v4_reset_done": True,
            "anti_collapse_v5_done": True,
        }

        try:
            with open(tmp, "wb") as stream:
                torch.save(payload, stream)
                stream.flush()
                os.fsync(stream.fileno())

            if (
                path.exists()
                and path.stat().st_size > MAX_CHECKPOINT_BYTES
            ):
                backup = path.with_name(
                    "checkpoint_legacy_backup.pth"
                )

                suffix = 1

                while backup.exists():
                    backup = path.with_name(
                        f"checkpoint_legacy_backup_{suffix}.pth"
                    )
                    suffix += 1

                os.replace(path, backup)

                print(
                    f"[AVISO] Checkpoint antigo preservado "
                    f"como {backup.name}"
                )

            os.replace(tmp, path)

            self._last_save_time = time.monotonic()
            size_mb = path.stat().st_size / (1024 * 1024)

            print(
                f"[SAVE] Checkpoint canonical v3 salvo: "
                f"{size_mb:.2f} MB | geração {self.generation}"
            )

        finally:
            if tmp.exists():
                try:
                    tmp.unlink()
                except OSError:
                    pass

        del payload

    def save_checkpoint(self, path=CHECKPOINT_PATH):
        with self._save_lock:
            return self._save_checkpoint_unlocked(path)

    def load_checkpoint(self, path=CHECKPOINT_PATH):
        path = Path(path)

        if not path.exists():
            return False

        size = path.stat().st_size

        if size > MAX_CHECKPOINT_BYTES:
            self._legacy_checkpoint = path

            print(
                f"[AVISO] Checkpoint antigo tem "
                f"{size / (1024 * 1024):.1f} MB."
            )

            print(
                "[AVISO] Não será carregado. "
                "O arquivo original será preservado."
            )

            return False

        try:
            data = torch.load(
                path,
                map_location="cpu",
                weights_only=True,
            )

            checkpoint_format = data.get("format")

            if checkpoint_format == "neural-chess-compact-v2":
                self.games_trained = int(
                    data.get("games_trained", 0)
                )

                self.generation = int(
                    data.get("generation", 0)
                )

                self.training_samples.clear()
                self._training_sample_seen = 0
                self._saved_optimizer_states = []

                self.rating = 400.0
                self.elo_history = []

                self._legacy_checkpoint = path

                print(
                    "[MIGRAÇÃO] Checkpoint v2 detectado. "
                    "Pesos e amostras antigos não serão usados."
                )

                print(
                    "[MIGRAÇÃO] Partidas e geração preservadas: "
                    f"{self.games_trained} partidas, "
                    f"geração {self.generation}."
                )

                print(
                    "[MIGRAÇÃO] Nova população canônica v3 "
                    "será inicializada. O arquivo v2 permanece intacto."
                )

                return False

            if checkpoint_format != CHECKPOINT_FORMAT:
                print(
                    "[AVISO] Formato de checkpoint não reconhecido; "
                    "iniciando população canônica nova."
                )
                return False

            self.model.load_state_dict(data["model"])

            self.population = []

            for saved_state in data["population"]:
                net = PolicyValueNet().to(self.device)
                net.load_state_dict(saved_state)
                self.population.append(net)

            if len(self.population) != POPULATION_SIZE:
                raise ValueError(
                    "Checkpoint não contém exatamente 20 indivíduos."
                )

            self.bongcloud_reset_done = bool(
                data.get("bongcloud_reset_done", False)
            )

            self._clean_v4_reset_done = True
            anti_collapse_applied = not data.get(
                "anti_collapse_v5_done",
                False,
            )

            if anti_collapse_applied:
                print(
                    "[ANTI-COLAPSO V5] Preservando indivíduo 0 "
                    "e diversificando indivíduos 1-19."
                )
                for index in range(1, POPULATION_SIZE):
                    self.mutate(
                        self.population[index],
                        scale=0.025,
                    )
                self.training_samples.clear()
                self._training_sample_seen = 0
                self._saved_optimizer_states = []
                self.last_loss = 0.0
                self.model.load_state_dict(
                    self.population[0].state_dict()
                )

            self.games_trained = int(
                data.get("games_trained", 0)
            )

            self.generation = int(
                data.get("generation", 0)
            )

            if data.get("rating_method") == RATING_METHOD:
                self.rating = float(
                    data.get("rating", 400.0)
                )

                self.elo_history = [
                    (float(a), float(b))
                    for a, b in data.get("elo_history", [])
                ][-500:]

            else:
                self.rating = 400.0
                self.elo_history = []

                print(
                    "[MIGRAÇÃO] Elo antigo descartado; "
                    "novo rating inicia em 400."
                )

            self.fitness = [
                float(x)
                for x in data.get("fitness", [])
            ][:POPULATION_SIZE]

            totals = data.get("result_totals", {})

            self.result_totals = {
                "white": int(totals.get("white", 0)),
                "draw": int(totals.get("draw", 0)),
                "black": int(totals.get("black", 0)),
            }

            self.recent_results = [
                str(x)
                for x in data.get("recent_results", [])
                if x in ("1-0", "1/2-1/2", "0-1")
            ][-100:]

            pending_totals = data.get(
                "pending_score_totals",
                [0.0] * POPULATION_SIZE,
            )

            pending_counts = data.get(
                "pending_game_counts",
                [0] * POPULATION_SIZE,
            )


            if (
                len(pending_totals) == POPULATION_SIZE
                and len(pending_counts) == POPULATION_SIZE
            ):
                self.pending_score_totals = [
                    float(x) for x in pending_totals
                ]

                self.pending_game_counts = [
                    int(x) for x in pending_counts
                ]

            self.next_genome_index = (
                int(data.get("next_genome_index", 0))
                % POPULATION_SIZE
            )

            if "last_benchmark_generation" in data:
                self.last_benchmark_generation = int(
                    data.get("last_benchmark_generation", 0)
                )
            else:
                self.last_benchmark_generation = (
                    self.generation // 5
                ) * 5

            self.benchmark_history = list(
                data.get("benchmark_history", [])
            )[-100:]

            self.training_samples = deque(maxlen=2048)

            if (
                not anti_collapse_applied
                and self.bongcloud_reset_done
                and data.get("bongcloud_reset_done", False)
            ):
                for (
                    packed,
                    action,
                    target,
                    policy_weight,
                    value_weight,
                ) in data.get("training_samples", [])[-2048:]:

                    bits = np.unpackbits(
                        np.frombuffer(packed, dtype=np.uint8),
                        count=768,
                    )

                    state = bits.astype(
                        np.float32
                    ).reshape(12, 8, 8)

                    self.training_samples.append((
                        state,
                        int(action),
                        float(target),
                        float(policy_weight),
                        float(value_weight),
                    ))

            self._training_sample_seen = (
                max(
                    len(self.training_samples),
                    int(data.get(
                        "training_sample_seen",
                        len(self.training_samples),
                    )),
                )
                if (
                    not anti_collapse_applied
                    and data.get("bongcloud_reset_done", False)
                )
                else 0
            )

            self.last_loss = float(
                data.get("last_loss", 0.0)
            )

            if (
                not anti_collapse_applied
                and data.get("bongcloud_reset_done", False)
            ):
                self._saved_optimizer_states = data.get(
                    "optimizer_states",
                    [],
                )

            self._last_save_time = time.monotonic()

            print(
                f"[CHECKPOINT] canonical v3 carregado: "
                f"{self.games_trained} partidas, "
                f"geração {self.generation}"
            )

            return True

        except Exception as exc:
            print(
                f"[AVISO] Falha ao carregar checkpoint: {exc}"
            )
            return False
