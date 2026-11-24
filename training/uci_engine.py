from __future__ import annotations

import contextlib
import os
import sys
import traceback
from pathlib import Path

import chess


def _write(message: str) -> None:
    sys.stdout.write(message + "\n")
    sys.stdout.flush()


def _parse_position(board: chess.Board, command: str) -> chess.Board:
    parts = command.strip().split()
    if len(parts) < 2 or parts.pop(0) != "position":
        raise ValueError("comando position inválido")

    if next(iter(parts),None) == "startpos":
        new_board = chess.Board()
        cursor = 2
    elif next(iter(parts),None) == "fen":
        try:
            moves_at = parts.index("moves", 2)
            fen_parts = parts[2:moves_at]
            cursor = moves_at + 1
        except ValueError:
            fen_parts = parts[2:]
            cursor = len(parts)
        if len(fen_parts) < 4:
            raise ValueError("FEN incompleto")
        new_board = chess.Board(" ".join(fen_parts))
    else:
        raise ValueError("position aceita startpos ou fen")

    if cursor < len(parts) and parts[cursor] == "moves":
        cursor += 1
    for token in parts[cursor:]:
        move = new_board.parse_uci(token)
        new_board.push(move)
    return new_board


def _compute_time_budget(board: chess.Board, command: str) -> float:
    """Calcula quanto tempo (em segundos) usar neste lance com base no relógio UCI."""
    tokens = command.split()
    params = {}
    i = 1
    while i < len(tokens):
        key = tokens[i]
        if key in ("wtime", "btime", "winc", "binc", "movetime", "movestogo") and i + 1 < len(tokens):
            try:
                params[key] = int(tokens[i + 1])
            except ValueError:
                pass
            i += 2
        else:
            i += 1

    if "movetime" in params:
        return max(0.3, min(6.0, (params["movetime"] / 1000.0) * 0.85))

    my_time_ms = params.get("wtime" if board.turn == chess.WHITE else "btime", 60000)
    my_inc_ms = params.get("winc" if board.turn == chess.WHITE else "binc", 0)

    my_time = max(0.5, my_time_ms / 1000.0)
    my_inc = max(0.0, my_inc_ms / 1000.0)

    if my_time < 8.0:
        return 0.35
    if my_time < 20.0:
        return min(1.2, my_time / 18.0 + my_inc * 0.5)

    moves_left = 18.0 if board.fullmove_number <= 25 else 15.0
    budget = (my_time / moves_left) + (my_inc * 0.75)
    return max(2.5, min(8.0, budget))


def _load_engine():
    os.chdir(Path(__file__).resolve().parent)
    with contextlib.redirect_stdout(sys.stderr):
        from brain import ChessBrain
        brain = ChessBrain()
        model = brain.make_benchmark_model()
        model.eval()
    return brain, model


def uci_loop() -> int:
    try:
        brain, model = _load_engine()
    except Exception:
        print("[UCI] Falha ao carregar checkpoint:\n" + traceback.format_exc(), file=sys.stderr)
        _write("bestmove 0000")
        return 1

    from brain import _model_move
    board = chess.Board()

    while True:
        line = sys.stdin.readline()
        if not line:
            break
        command = line.strip()
        if not command:
            continue
        head = command.split(maxsplit=1).pop(0)

        try:
            if head == "uci":
                _write("id name XMTCore")
                _write("id author Xamito")
                _write("option name Move Overhead type spin default 100 min 0 max 5000")
                _write("option name Threads type spin default 1 min 1 max 32")
                _write("option name Hash type spin default 16 min 1 max 1024")
                _write("option name Ponder type check default false")
                _write("option name SyzygyPath type string default <empty>")
                _write("uciok")
            elif head == "isready":
                _write("readyok")
            elif head == "ucinewgame":
                board = chess.Board()
            elif head == "position":
                board = _parse_position(board, command)
            elif head == "go":
                budget = _compute_time_budget(board, command)
                move = _model_move(model, board, temperature=0.0, time_limit=budget)
                _write("bestmove " + (move.uci() if move is not None else "0000"))
            elif head == "stop":
                continue
            elif head == "quit":
                break
            elif head in ("setoption", "ponderhit", "register", "debug"):
                continue
            else:
                continue
        except Exception as exc:
            print(f"[UCI] Erro em '{command}': {exc}", file=sys.stderr)
            if head == "go":
                _write("bestmove 0000")

    del model, brain
    return 0


if __name__ == "__main__":
    raise SystemExit(uci_loop())
