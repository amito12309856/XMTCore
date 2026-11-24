from __future__ import annotations

import gc
import math
import os
import random
import sys
import time
from pathlib import Path

import chess
import chess.engine
import chess.pgn
import numpy as np
import torch
import torch.nn.functional as F

from brain import ChessBrain, _encode_board_array, move_to_action


def _find_teacher_engine() -> Path | None:
    """Procura automaticamente por uma engine UCI forte nas pastas do Arena / PC."""
    candidates = [
        Path(r"C:\Program Files (x86)\Arena\Engines"),
        Path(r"C:\Program Files\Arena\Engines"),
        Path(r"C:\Arena\Engines"),
        Path.home() / "Downloads",
    ]
    preferred = ("stockfish", "hermann", "anmon", "rybka", "sos", "ruffian")
    found = []
    for base in candidates:
        if not base.exists():
            continue
        for exe in base.rglob("*.exe"):
            low = exe.name.lower()
            if any(k in low for k in preferred):
                found.append(exe)
    if not found:
        return None
    found.sort(key=lambda p: next((i for i, k in enumerate(preferred) if k in p.name.lower()), 99))
    return found.pop(0)


def _sample_positions_from_pgn(pgn_path: Path, target_positions: int = 12000) -> list[chess.Board]:
    """Extrai posições ricas de meio-jogo e final de partidas reais."""
    boards: list[chess.Board] = []
    if not pgn_path.is_file():
        return boards

    with open(pgn_path, "r", encoding="utf-8", errors="ignore") as f:
        while len(boards) < target_positions:
            game = chess.pgn.read_game(f)
            if game is None:
                break
            board = game.board()
            ply = 0
            for move in game.mainline_moves():
                board.push(move)
                ply += 1
                # Pega posições de meio-jogo e final (a partir do lance 10)
                if ply >= 10 and not board.is_game_over():
                    if random.random() < 0.25:
                        boards.append(board.copy(stack=False))
                        if len(boards) >= target_positions:
                            break
    return boards


def run_distillation(
    teacher_exe: str | None = None,
    pgn_file: str = "mestres_20k.pgn",
    num_positions: int = 50000,
    think_time: float = 0.08,
    batch_size: int = 512,
    epochs: int = 16,
    lr: float = 6e-5,
) -> None:
    engine_path = Path(teacher_exe) if teacher_exe else _find_teacher_engine()
    if engine_path is None or not engine_path.is_file():
        print("[ERRO] Nenhuma engine UCI encontrada automaticamente.")
        print("Passe o caminho do .exe da engine como 1º argumento, ex:")
        print(r'  .\.venv\Scripts\python.exe distill_engine.py "C:\Program Files (x86)\Arena\Engines\Hermann\Hermann28.exe"')
        return

    print(f"[PROFESSOR] Usando engine UCI: {engine_path}")
    pgn_path = Path(pgn_file)
    print(f"[DATASET] Extraindo {num_positions} posições táticas de '{pgn_path.name}'...")
    positions = _sample_positions_from_pgn(pgn_path, target_positions=num_positions)
    if not positions:
        print(f"[ERRO] Não foi possível extrair posições de '{pgn_path}'.")
        return

    states = []
    actions = []
    values = []

    t0 = time.perf_counter()
    print(f"[ANÁLISE] Consultando o Professor em {len(positions)} posições ({think_time}s/pos)...")
    with chess.engine.SimpleEngine.popen_uci(str(engine_path)) as teacher:
        limit = chess.engine.Limit(time=think_time)
        for idx, board in enumerate(positions, start=1):
            info = teacher.analyse(board, limit)
            pv = info.get("pv")
            score_obj = info.get("score")
            if not pv or score_obj is None:
                continue

            best_move = next(iter(pv))
            pov_score = score_obj.pov(board.turn)
            if pov_score.is_mate():
                mate_in = pov_score.mate()
                z = 1.0 if (mate_in is not None and mate_in > 0) else -1.0
            else:
                cp = pov_score.score(mate_score=10000) or 0
                z = math.tanh(float(cp) / 380.0)

            states.append(_encode_board_array(board))
            actions.append(move_to_action(best_move, board.turn))
            values.append(z)

            if idx % 1000 == 0:
                elapsed = time.perf_counter() - t0
                print(f"  -> {idx}/{len(positions)} posições rotuladas ({elapsed:.1f}s)...")

    if not states:
        print("[ERRO] Nenhuma análise válida retornada pela engine.")
        return

    X = np.stack(states)
    Y_act = np.asarray(actions, dtype=np.int64)
    Y_val = np.asarray(values, dtype=np.float32)
    del positions, states, actions, values
    gc.collect()

    brain = ChessBrain()
    model = brain.model
    device = brain.device
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    n_samples = len(X)
    print(f"\n[TREINO RTX] Treinando {n_samples} posições destiladas na {device}...")
    for epoch in range(1, epochs + 1):
        perm = np.random.permutation(n_samples)
        total_loss = 0.0
        steps = 0
        for start in range(0, n_samples, batch_size):
            idx = perm[start : start + batch_size]
            bx = torch.from_numpy(X[idx]).to(device)
            ba = torch.from_numpy(Y_act[idx]).to(device)
            bv = torch.from_numpy(Y_val[idx]).to(device)

            optimizer.zero_grad(set_to_none=True)
            logits, val_pred = model(bx)
            policy_loss = F.cross_entropy(logits, ba)
            value_loss = F.mse_loss(val_pred, bv)
            loss = policy_loss + 0.65 * value_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()

            total_loss += float(loss.item())
            steps += 1

        print(f"  [ÉPOCA {epoch}/{epochs}] Loss médio: {total_loss / max(1, steps):.4f}")

    brain.games_trained += max(1, n_samples // 60)
    brain.save_checkpoint()
    print("\n[CONCLUÍDO] Destilação concluída e salva no checkpoint.pth!")


if __name__ == "__main__":
    exe_arg = sys.argv.pop(1) if len(sys.argv) > 1 else None
    pos_arg = int(sys.argv.pop(1)) if len(sys.argv) > 1 else 50000
    run_distillation(teacher_exe=exe_arg, num_positions=pos_arg)
