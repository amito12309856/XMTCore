#!/usr/bin/env python3
"""Calibration match: XMTCore vs Stockfish (UCI_LimitStrength). Requires python-chess.

  python tools/match.py --engine ./XMTCore --stockfish /usr/games/stockfish --elo 1500 --games 8
XMTCore plays with a 20s+0.2s clock; Stockfish with a fixed --sf-movetime per move.
Colors alternate. Every engine move is checked for legality. Prints the score of XMTCore.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor

import chess
import chess.engine


def play(job):
    i, a = job
    x = chess.engine.SimpleEngine.popen_uci(a.engine)
    x.configure({"UseNN": not a.no_nn, "Move Overhead": 20, "Threads": a.threads})
    s = chess.engine.SimpleEngine.popen_uci(a.stockfish)
    s.configure({"UCI_LimitStrength": True, "UCI_Elo": a.elo, "Threads": 1})
    board, white_x = chess.Board(), i % 2 == 0
    while not board.is_game_over(claim_draw=True) and len(board.move_stack) < 300:
        mine = (board.turn == chess.WHITE) == white_x
        lim = (chess.engine.Limit(white_clock=20, black_clock=20, white_inc=0.2, black_inc=0.2)
               if mine else chess.engine.Limit(time=a.sf_movetime))
        mv = (x if mine else s).play(board, lim).move
        assert mv in board.legal_moves, (board.fen(), mv)
        board.push(mv)
    x.quit(); s.quit()
    r = {"1-0": 1.0, "0-1": 0.0}.get(board.result(claim_draw=True), 0.5)
    return r if white_x else 1.0 - r


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", required=True)
    ap.add_argument("--stockfish", required=True)
    ap.add_argument("--elo", type=int, default=1500)
    ap.add_argument("--games", type=int, default=8)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--sf-movetime", type=float, default=0.1)
    ap.add_argument("--no-nn", action="store_true")
    a = ap.parse_args()
    with ProcessPoolExecutor(a.workers) as ex:
        res = list(ex.map(play, [(i, a) for i in range(a.games)]))
    print(f"XMTCore {sum(res)}/{a.games} vs Stockfish UCI_Elo={a.elo} ({a.sf_movetime}s/move)  {res}")
