#!/usr/bin/env python3
"""
export_model.py - exporta a PolicyValueNet do checkpoint.pth para o motor nativo XMTCore.

Gera:
  * xmtcore.nn          - pesos float32 em binario leve (~0.1 MB), lidos direto pelo C++
  * nn_reference.txt    - posicoes + saidas de referencia calculadas pelo PyTorch real
                          (para validar o C++:  XMTCore.exe --nn-test nn_reference.txt)
  * (opcional) --onnx   - xmtcore.onnx, so se voce quiser usar ONNX Runtime/outra ferramenta

Uso (na pasta do projeto, onde estao brain.py e checkpoint.pth):
  python export_model.py
  python export_model.py --checkpoint checkpoint.pth --best-index 0 --out xmtcore.nn
"""
import argparse
import random
import struct
import sys
from pathlib import Path

import numpy as np

MAGIC = b"XMTN"
VERSION = 1


def load_population_net(path, best_index, use_model_key):
    """Reproduz o que o uci_engine.py faz hoje: population[best_index] (best_index=0 por padrao)."""
    import torch

    data = torch.load(path, map_location="cpu", weights_only=True)
    fmt = data.get("format")
    if fmt != "neural-chess-canonical-v3":
        sys.exit(f"[ERRO] formato de checkpoint inesperado: {fmt!r}")
    if use_model_key:
        state = data["model"]
        origin = "chave 'model'"
    else:
        pop = data["population"]
        if not 0 <= best_index < len(pop):
            sys.exit(f"[ERRO] best_index {best_index} fora de 0..{len(pop) - 1}")
        state = pop[best_index]
        origin = f"population[{best_index}]"
    from brain import PolicyValueNet  # mesma classe usada no treino/lichess

    net = PolicyValueNet()
    net.load_state_dict(state)  # fp16 do checkpoint -> fp32 (identico ao que roda hoje)
    net.eval()
    return net, data, origin


def state_to_numpy(net):
    return {k: v.detach().cpu().numpy().astype(np.float32) for k, v in net.state_dict().items()}


def export_binary(state, meta, out_path):
    """state: dict nome -> np.ndarray (float32) da PolicyValueNet. meta: dict com generation/games_trained."""
    C = state["features.0.weight"].shape[0]
    E = state["policy_features.weight"].shape[0]
    H = state["value_head.0.weight"].shape[0]
    if (C, E, H) != (32, 16, 64):
        sys.exit(
            f"[ERRO] arquitetura {C}/{E}/{H} difere de 32/16/64 compilada no C++. "
            "Ajuste Network::C/E/H em nn.h e recompile."
        )

    def conv_hwio(name):  # torch (cout, cin, ky, kx) -> (ky, kx, cin, cout): cout contiguo
        return np.ascontiguousarray(state[name].transpose(2, 3, 1, 0))

    parts = [
        conv_hwio("features.0.weight"), state["features.0.bias"],
        conv_hwio("features.2.weight"), state["features.2.bias"],
        conv_hwio("features.4.weight"), state["features.4.bias"],
        state["policy_features.weight"].reshape(E, C), state["policy_features.bias"],
        state["from_head.weight"], state["to_head.weight"],
        state["value_head.0.weight"], state["value_head.0.bias"],
        state["value_head.2.weight"].reshape(H),
        state["promotion_bias"],
        state["value_head.2.bias"],
    ]
    with open(out_path, "wb") as f:
        f.write(MAGIC)
        f.write(struct.pack("<8I", VERSION, 12, C, E, H,
                            int(meta.get("generation", 0)), int(meta.get("games_trained", 0)), 0))
        for p in parts:
            f.write(np.ascontiguousarray(p, dtype="<f4").tobytes())
    return Path(out_path).stat().st_size


def reference_fens(n_random, seed):
    import chess

    rng = random.Random(seed)
    fens = [
        chess.STARTING_FEN,
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R b KQkq - 0 1",
        "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
        "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
        "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 b - - 0 10",
    ]
    for _ in range(n_random):  # posicoes de partidas aleatorias (brancas e pretas a jogar)
        b = chess.Board()
        for _ in range(rng.randint(4, 70)):
            moves = list(b.legal_moves)
            if not moves:
                break
            b.push(rng.choice(moves))
        if not b.is_game_over():
            fens.append(b.fen())
    return fens


def write_reference(net, path, n_random):
    import chess
    import torch
    from brain import _encode_board_array, move_to_action, PROMO_INDEX

    lines = []
    torch.set_grad_enabled(False)
    for fen in reference_fens(n_random, seed=12345):
        b = chess.Board(fen)
        x = torch.from_numpy(_encode_board_array(b)).unsqueeze(0)
        logits, value = net(x)
        logits = logits[0]
        promo = net.promotion_bias.detach()
        scored = []
        for m in b.legal_moves:
            a = move_to_action(m, b.turn)
            scored.append((float(logits[a] + promo[PROMO_INDEX.get(m.promotion, 0)]), m.uci()))
        scored.sort(reverse=True)
        top = ",".join(f"{u}:{s:.6f}" for s, u in scored[:10])
        lines.append(f"{fen}|{float(value[0]):.6f}|{top}")
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")
    return len(lines)


def export_onnx(net, path):
    import torch

    class Wrap(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, x):
            return self.m(x)

    dummy = torch.zeros(1, 12, 8, 8)
    torch.onnx.export(Wrap(net), dummy, path, input_names=["board"], output_names=["policy", "value"],
                      dynamic_axes={"board": {0: "batch"}, "policy": {0: "batch"}, "value": {0: "batch"}},
                      opset_version=17)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="checkpoint.pth")
    ap.add_argument("--best-index", type=int, default=0, help="indice na population (o uci_engine usa 0)")
    ap.add_argument("--use-model-key", action="store_true", help="usa data['model'] em vez de population[i]")
    ap.add_argument("--out", default="xmtcore.nn")
    ap.add_argument("--reference", default="nn_reference.txt")
    ap.add_argument("--n-reference", type=int, default=40)
    ap.add_argument("--onnx", default=None, help="caminho opcional para tambem gerar .onnx")
    args = ap.parse_args()

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    sys.path.insert(0, str(Path.cwd()))
    net, data, origin = load_population_net(args.checkpoint, args.best_index, args.use_model_key)
    size = export_binary(state_to_numpy(net), data, args.out)
    print(f"[OK] {args.out}: {size / 1024:.1f} KB  (origem: {origin}, geracao {data.get('generation')}, "
          f"{data.get('games_trained')} partidas)")
    n = write_reference(net, args.reference, args.n_reference)
    print(f"[OK] {args.reference}: {n} posicoes de referencia (saidas do PyTorch)")
    if args.onnx:
        export_onnx(net, args.onnx)
        print(f"[OK] {args.onnx}")
    print("Proximo passo:  XMTCore.exe --nn-test nn_reference.txt xmtcore.nn")


if __name__ == "__main__":
    main()
