from __future__ import annotations

import gc
import sys
import time
from pathlib import Path

import chess
import chess.pgn
import numpy as np
import torch
import torch.nn.functional as F

from brain import ChessBrain, _encode_board_array, move_to_action


def _train_chunk(model, optimizer, device, states, actions, outcomes, batch_size=512, epochs=3):
    X = np.stack(states)
    Y_act = np.asarray(actions, dtype=np.int64)
    Y_val = np.asarray(outcomes, dtype=np.float32)
    n_samples = len(X)

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
            logits, value = model(bx)
            policy_loss = F.cross_entropy(logits, ba)
            value_loss = F.mse_loss(value, bv)
            loss = policy_loss + 0.5 * value_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()

            total_loss += float(loss.item())
            steps += 1

        print(f"    [ÉPOCA {epoch}/{epochs}] Loss médio: {total_loss / max(1, steps):.4f}")

    del X, Y_act, Y_val
    gc.collect()


def train_from_pgn(
    pgn_path: str,
    max_games: int = 25000,
    chunk_games: int = 4000,
    batch_size: int = 512,
    epochs: int = 3,
    lr: float = 1e-4,
) -> None:
    path = Path(pgn_path)
    if not path.is_file():
        print(f"[ERRO] Arquivo PGN não encontrado: {path}")
        return

    brain = ChessBrain()
    model = brain.model
    device = brain.device
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    states = []
    actions = []
    outcomes = []

    print(f"[PGN] Lendo até {max_games} partidas de '{path.name}' em blocos de {chunk_games} (baixo uso de RAM)...")
    t0 = time.perf_counter()
    games_read = 0
    chunk_idx = 0

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        while games_read < max_games:
            game = chess.pgn.read_game(f)
            if game is None:
                break

            result = game.headers.get("Result", "*")
            if result == "1-0":
                white_z = 1.0
            elif result == "0-1":
                white_z = -1.0
            elif result == "1/2-1/2":
                white_z = 0.0
            else:
                continue

            board = game.board()
            for move in game.mainline_moves():
                mover = board.turn
                states.append(_encode_board_array(board))
                actions.append(move_to_action(move, mover))
                outcomes.append(white_z if mover == chess.WHITE else -white_z)
                board.push(move)

            games_read += 1

            if games_read % chunk_games == 0:
                chunk_idx += 1
                print(f"\n[BLOCO {chunk_idx}] {games_read} partidas lidas | Treinando {len(states)} posições na {device}...")
                _train_chunk(model, optimizer, device, states, actions, outcomes, batch_size=batch_size, epochs=epochs)
                states.clear()
                actions.clear()
                outcomes.clear()
                gc.collect()
                brain.games_trained += chunk_games
                brain.save_checkpoint()

    if states:
        chunk_idx += 1
        rem = games_read % chunk_games
        print(f"\n[BLOCO {chunk_idx} FINAL] Treinando {len(states)} posições restantes na {device}...")
        _train_chunk(model, optimizer, device, states, actions, outcomes, batch_size=batch_size, epochs=epochs)
        states.clear()
        actions.clear()
        outcomes.clear()
        gc.collect()
        brain.games_trained += rem if rem > 0 else games_read
        brain.save_checkpoint()

    print(f"\n[CONCLUÍDO] {games_read} partidas de Mestres treinadas em {time.perf_counter() - t0:.1f}s!")


if __name__ == "__main__":
    pgn_file = sys.argv.pop(1) if len(sys.argv) > 1 else "mestres_20k.pgn"
    limit = int(sys.argv.pop(1)) if len(sys.argv) > 1 else 25000
    train_from_pgn(pgn_file, max_games=limit)
