#pragma once
#include <atomic>
#include <chrono>
#include <cstdint>
#include <memory>
#include <vector>

#include "chess.h"
#include "nn.h"

namespace xmt {

constexpr int MAX_PLY = 128;
constexpr int MATE = 32000;
constexpr int MATE_BOUND = MATE - 256;
constexpr int INF = 32001;

enum TTFlag : uint8_t {
  TT_NONE = 0,
  TT_EXACT = 1,
  TT_ALPHA = 2,  // upper bound (fail-low)
  TT_BETA = 3    // lower bound (fail-high)
};

struct TTData {
  Move move;
  int score;
  int depth;
  uint8_t flag;
};

// Lock-free transposition table shared by all search threads. Each 16-byte entry stores
// (key ^ data, data); a torn read fails the key check and is treated as a miss.
class TranspositionTable {
 public:
  void resize(size_t mb);
  void clear();
  void new_search() { ++age; }
  bool probe(uint64_t key, TTData& out) const;
  void store(uint64_t key, Move move, int score, int depth, uint8_t flag);
  int hashfull() const;

 private:
  struct Entry {
    std::atomic<uint64_t> keyx;
    std::atomic<uint64_t> data;
  };
  std::unique_ptr<Entry[]> table;
  size_t mask = 0;
  size_t count = 0;
  uint8_t age = 0;
};

struct Limits {
  int64_t wtime = -1, btime = -1, winc = 0, binc = 0, movetime = -1;
  int movestogo = 0;
  int depth = 0;
  uint64_t nodes = 0;
  bool infinite = false;
  bool ponder = false;
};

struct SearchOptions {
  bool useNN = true;
  int nnPly = 2;        // network policy orders moves up to this ply from the root
  int nnMargin = 15;    // cp: root moves within this margin of the best are re-ranked by the network
  int nnValueCp = 30;   // weight (cp) of the Value Head in the root tiebreak
  int nnPolicyCp = 20;  // weight (cp) of the Policy Head in the root tiebreak
  int moveOverhead = 100;
};

struct RootMove {
  Move move = NO_MOVE;
  int score = -INF;
  bool exact = false;
  float prob = 0.f;
};

class Searcher;

// One search thread. Worker 0 is the main thread (time management, output, final move
// selection); the others are Lazy SMP helpers that only populate the shared TT.
class Worker {
 public:
  Worker(int id, Searcher* mgr);
  void reset();
  Move think(const Position& root, const Limits& lim);
  Move best_ponder() const { return ponderMove; }

  SearchOptions opts;
  const Network* net = nullptr;
  std::atomic<uint64_t> pubNodes{0};

 private:
  struct NNCacheEntry {
    uint64_t key = 0;
    float value = 0;
    uint16_t n = 0;
    uint8_t prob[256];
  };

  int search(int depth, int alpha, int beta, int ply, bool doNull);
  int qsearch(int alpha, int beta, int ply);
  int search_root(int depth, int alpha, int beta);
  int score_move(Move m, Move ttMove, int ply, int prior) const;
  const NNCacheEntry* nn_priors(const Move* list, int n);
  bool stopped() const;
  void check_time();
  int64_t elapsed_ms() const;
  void print_info(int depth, int score, int64_t ms);
  void hash_update_pv(int ply, Move m);

  const int id;
  Searcher* mgr;
  Position pos;

  std::vector<NNCacheEntry> nnCache;
  std::vector<float> logitBuf;

  int history[2][64][64];
  Move counterMoves[12][64]{};
  Move lastMove[MAX_PLY + 2]{};
  int lastPiece[MAX_PLY + 2]{};
  Move killers[MAX_PLY + 2][2];
  Move pvTable[MAX_PLY + 2][MAX_PLY + 2];
  int pvLen[MAX_PLY + 2];
  int lmr[64][64];

  std::vector<RootMove> root, completed;
  Move iterBestMove = NO_MOVE;
  Move ponderMove = NO_MOVE;
  Move completedPonder = NO_MOVE;
  int rootDepth = 0, seldepth = 0;
  uint64_t nodes = 0, nodeLimit = 0, nnEvals = 0;
  std::chrono::steady_clock::time_point startTime;
  int64_t softMs = 0, hardMs = 0;
  bool timed = false;
  bool wasPondering = false;
};

// Search manager: owns the shared TT and the worker threads (Lazy SMP).
class Searcher {
 public:
  Searcher();
  void set_hash_mb(int mb);
  void set_threads(int n);
  void new_game();
  void set_network(const Network* n) { net = n; }

  SearchOptions opts;
  std::atomic<bool> stopFlag{false};
  std::atomic<bool> ponderFlag{false};
  void ponder_hit() { ponderFlag.store(false, std::memory_order_release); }
  TranspositionTable tt;

  // Blocking. Runs all threads, prints "info" lines, returns the best move.
  Move think(const Position& root, const Limits& lim);

  uint64_t total_nodes() const;  // summed over all threads
  uint64_t last_nodes() const { return lastNodes; }
  Move last_ponder() const { return lastPonder; }

 private:
  std::vector<std::unique_ptr<Worker>> workers;
  const Network* net = nullptr;
  uint64_t lastNodes = 0;
  Move lastPonder = NO_MOVE;
  int threads = 1;
};

}  // namespace xmt
