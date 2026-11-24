# XMTCore

XMTCore is a UCI chess engine written in C++17 that combines a multi-threaded alpha-beta search with a
small convolutional policy/value network trained in PyTorch. The search has no external dependencies
(no ONNX Runtime, no DLLs); the network is exported to a flat binary file (`xmtcore.nn`, 98.5 KB) and
evaluated by a self-contained C++ implementation.

Lichess: https://lichess.org/@/XMTCore

## Specifications

| Item | Value |
|---|---|
| Protocol | UCI |
| Language | C++17 (search, move generation, network inference); Python 3 / PyTorch (training, export) |
| Board representation | Bitboards, magic bitboards for sliders |
| Search | Negamax, PVS, iterative deepening, aspiration windows, Lazy SMP (1-64 threads) |
| Transposition table | Shared, lock-free, 16-byte entries, flags EXACT / ALPHA / BETA |
| Evaluation | PeSTO piece-square tables + pawn structure, bishop pair, rook files, king shield |
| Network | `PolicyValueNet`, input `(12, 8, 8)`, 3 x conv3x3 (32 channels), policy over 4096 from-to pairs, scalar value |
| Network file | `xmtcore.nn`, float32, 98.5 KB |
| Current network | Generation 1002, ~129,591 training games (self-play, grandmaster PGN, tactical distillation) |
| Not supported | Chess960 |

## Architecture

```
  Training pipeline (Python / PyTorch)                 Engine (C++17, no dependencies)
  ------------------------------------                 -------------------------------
  self-play, GM PGN, distillation
  train_pgn.py, distill_engine.py                      +----------------------------------------+
            |                                          |  UCI loop (main.cpp)                   |
            v                                          |    go / stop / setoption / position    |
     checkpoint.pth                                    +--------------------+-------------------+
            |                                                               |
            | export_model.py                           +-------------------v-------------------+
            v                                           |  Searcher (Lazy SMP manager)          |
  +-------------------+     weights/xmtcore.nn          |    shared TT (lock-free)              |
  | xmtcore.nn        +-------------------------------->|    Worker 0 : time, output, root pick |
  | nn_reference.txt  |                                 |    Worker 1..N-1 : helper searches    |
  +-------------------+                                 +----+-----------------------+----------+
                                                             |                       |
                                                 +-----------v---------+  +----------v----------+
                                                 | Classical eval      |  | Network (nn.cpp)    |
                                                 | PeSTO + structure   |  | policy: move order  |
                                                 | used at every node  |  | value: root tiebreak|
                                                 +---------------------+  +---------------------+
```

## Search

| Component | Details |
|---|---|
| Core | Negamax with Principal Variation Search, iterative deepening, aspiration windows (from depth 5) |
| Parallelism | Lazy SMP. Threads share the transposition table and search the root with staggered depths. Each thread has its own history, killers, and network cache. Reported `nodes` and `nps` are summed over all threads |
| Transposition table | Single shared table, bound flags `EXACT`, `ALPHA` (upper), `BETA` (lower), mate-score normalization |
| Pruning | Null-move pruning, reverse futility pruning, razoring, futility pruning, late-move pruning, SEE-based pruning of losing captures |
| Reductions / extensions | Late move reductions, check extension, singular extensions |
| Quiescence | Captures and queen promotions with SEE and delta pruning; full evasion generation when in check |
| Move ordering | Hash move, winning captures (SEE), killers, countermove heuristic, history heuristic, network policy prior |
| Draws | Fifty-move rule, repetition (including game history), insufficient material |
| Time management | Soft/hard limits derived from `wtime`/`btime`/`winc`/`binc`/`movestogo`, `Move Overhead` |

## Evaluation and neural integration

The classical evaluation (PeSTO tables with midgame/endgame interpolation, passed/isolated/doubled pawns,
bishop pair, rooks on open and semi-open files, king pawn shield, mop-up term) is used at every node. The
network is not evaluated at leaf nodes: one forward pass costs roughly 1.4 M multiply-accumulates, which would
reduce node throughput by two orders of magnitude.

| Use | Where | Controlled by |
|---|---|---|
| Policy Head as move-ordering prior | Root and the first `NNPly` plies, cached by Zobrist key (`NNCache`) | `NNPly` |
| Value Head + Policy Head as tiebreak | Root moves whose search score is within `NNMargin` cp of the best | `NNMargin`, `NNValueCp`, `NNPolicyCp` |

The tiebreak never selects a move outside the margin, so material safety is determined by the search.

### Network

| Layer | Shape |
|---|---|
| Input | `(12, 8, 8)`, side to move in planes 0-5, opponent in planes 6-11; squares mirrored (`square ^ 56`) when Black is to move |
| Trunk | 3 x (conv 3x3, 32 channels, ReLU) |
| Policy Head | conv 1x1 (32 to 16); source and destination projections (16x16); `src . dst / 4` gives 4096 logits indexed `from * 64 + to`; plus a 5-element `promotion_bias` |
| Value Head | global average pool, linear 32 to 64, ReLU, linear 64 to 1, tanh |

### `xmtcore.nn` format

Little-endian. Header: magic `XMTN`, then eight `uint32`: version (1), input channels (12), channels (32),
embedding (16), value hidden size (64), generation, games trained, reserved. Followed by float32 tensors in
this order: conv1 weights `(3,3,12,32)`, conv1 bias, conv2 weights `(3,3,32,32)`, conv2 bias, conv3 weights,
conv3 bias, policy-feature weights `(16,32)`, bias `(16)`, `from_head` `(16,16)`, `to_head` `(16,16)`, value
layer 1 weights `(64,32)`, bias `(64)`, value layer 2 weights `(64)`, `promotion_bias` `(5)`, value layer 2
bias. Convolution weights are stored as `(ky, kx, in, out)`.

## Repository layout

```
src/            C++17 engine sources
training/       brain.py, train_pgn.py, distill_engine.py, export_model.py, requirements.txt
weights/        xmtcore.nn, nn_reference.txt
tools/          match.py (calibration matches against Stockfish)
Makefile        Linux / MinGW build
build.bat       Windows MSVC build
build_mingw.bat Windows MinGW-w64 build
```

## Build

The network file must be next to the executable (`xmtcore.nn`) or set with the `NNPath` UCI option. Without
it the engine runs with the classical evaluation only.

### Linux / macOS / MinGW (make)

```
make                       # g++ -O3 -march=native -flto
make ARCH=x86-64-v3        # portable AVX2 build (use x86-64-v2 for older CPUs)
make test                  # perft(5) check
cp weights/xmtcore.nn .
```

Plain `g++`:

```
g++ -O3 -std=c++17 -march=native -flto -DNDEBUG -o XMTCore src/*.cpp -pthread
```

### Windows, MSVC

Open "x64 Native Tools Command Prompt for VS 2022" in the repository root:

```
build.bat              # AVX2 build        -> XMTCore.exe
build.bat generic      # baseline x64 build -> XMTCore-generic.exe
copy weights\xmtcore.nn .
```

### Windows, MinGW-w64 (MSYS2)

```
build_mingw.bat              # x86-64-v3 (AVX2) -> XMTCore.exe
build_mingw.bat generic      # baseline x86-64  -> XMTCore-generic.exe
```

## Validation

```
XMTCore --perft 5
XMTCore --perft 6
XMTCore --nn-test weights/nn_reference.txt weights/xmtcore.nn
XMTCore --bench 12 1 64
XMTCore --bench 20 8 512
```

| Command | Expected |
|---|---|
| `--perft 5` | `perft(5) = 4865609` |
| `--perft 6` | `perft(6) = 119060324` |
| `--nn-test <ref> <nn>` | `max\|dValue\|` below 2e-3, `max\|dLogit\|` below 2e-2, `errors=0`, `OK` |
| `--bench <depth> <threads> <hash_mb>` | Fixed-depth search of six positions; prints total nodes and aggregate nps |
| `--version` | `XMTCore 1.3.0` |

`nn_reference.txt` is generated by `export_model.py` from the PyTorch model (see Training). `--nn-test`
compares the C++ forward pass against those outputs, including positions with Black to move.

## UCI options

| Option | Type | Default | Range | Description |
|---|---|---|---|---|
| `Hash` | spin | 64 | 1-4096 | Transposition table size in MB |
| `Threads` | spin | 1 | 1-64 | Number of search threads (Lazy SMP) |
| `Move Overhead` | spin | 100 | 0-5000 | Time (ms) reserved per move for communication latency |
| `Clear Hash` | button | | | Clears the transposition table and per-thread heuristics |
| `UseNN` | check | true | | Enables the network (move ordering and root tiebreak) |
| `NNPath` | string | empty | | Path to the network file; default is `xmtcore.nn` next to the executable |
| `NNPly` | spin | 2 | 0-6 | Plies from the root at which the policy is used for move ordering |
| `NNMargin` | spin | 15 | 0-100 | Root tiebreak margin in centipawns; 0 disables the tiebreak |
| `NNValueCp` | spin | 30 | 0-300 | Tiebreak weight of the Value Head (centipawns at value = 1) |
| `NNPolicyCp` | spin | 20 | 0-300 | Tiebreak weight of the Policy Head (centipawns at probability = 1) |

Unknown options are ignored.

## Training and export

```
cd training
python -m venv .venv && .venv/Scripts/activate        # Linux: source .venv/bin/activate
pip install -r requirements.txt

python train_pgn.py masters.pgn 25000                 # supervised training from PGN
python distill_engine.py /path/to/teacher_engine 50000 masters.pgn
python export_model.py                                # writes ../weights/xmtcore.nn and nn_reference.txt
```

`train_pgn.py` and `distill_engine.py` update `checkpoint.pth` in the working directory. `export_model.py`
loads `population[0]` (the individual used for play; `--best-index` selects another) and writes the binary
weights together with reference outputs computed by the PyTorch model on 40+ positions. After every export:

```
XMTCore --nn-test weights/nn_reference.txt weights/xmtcore.nn
```

The exporter assumes the architecture compiled into the engine (`Network::C/E/H` in `src/nn.h`); it aborts if
the checkpoint differs. `export_model.py --onnx model.onnx` additionally writes an ONNX file for use with other
tools; the engine does not use it.

## lichess-bot

In `config.yml` of [lichess-bot](https://github.com/lichess-bot-devs/lichess-bot):

```yaml
engine:
  dir: "C:/engines/XMTCore"
  name: "XMTCore.exe"
  protocol: "uci"
  ponder: false
  uci_options:
    Threads: 4
    Hash: 512
    Move Overhead: 150
```

`xmtcore.nn` must be located in `dir`. Configure the opening book / explorer in lichess-bot; the engine
itself has no opening book.

## Benchmarks

Measured by the author on a Windows 11 x64 machine with 16 GB RAM.

| Measurement | Result |
|---|---|
| `perft(5)` | 4,865,609 nodes, 34.1 Mnps |
| `perft(6)` | 119,060,324 nodes |
| Search speed, 1 thread | ~3.5 Mnps |
| Search speed, 8 threads (Lazy SMP, Hash 512 MB) | ~25-26 Mnps aggregate; depth 20-27 in middlegame and endgame positions |
| C++ vs PyTorch parity, 46 reference positions | `max\|dValue\|` = 0.000001, `max\|dLogit\|` = 0.000008, 0 errors |
| vs Stockfish 16, `UCI_Elo` 1500 | 16 / 16 |
| vs Stockfish 16, `UCI_Elo` 2200 | 15 / 16 |
| Lichess (`@XMTCore`), first run with the C++ search | 19 wins, 0 draws, 0 losses (+426 rating) |

The Stockfish matches use `UCI_LimitStrength` with 0.1 s per Stockfish move (XMTCore on a clock with increment),
colors alternating. `UCI_Elo` is calibrated by Stockfish for longer time controls, so the figures are relative
and the sample is small (16 games per level). The procedure is reproducible with:

```
python tools/match.py --engine ./XMTCore --stockfish /path/to/stockfish --elo 1500 --games 8
```

The Lichess result is a rating-pool outcome and depends on the opponent distribution.

## Credits

- PeSTO piece-square tables and phase weights: Ronald Friederich (Rofchade).
- [python-chess](https://github.com/niklasf/python-chess), [PyTorch](https://pytorch.org/), and
  [lichess-bot](https://github.com/lichess-bot-devs/lichess-bot) are used in training, testing, and deployment.

## License

GNU General Public License v3.0 (GPL-3.0). See `LICENSE`.





