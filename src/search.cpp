#include "search.h"

#include <cmath>
#include <cstdio>
#include <mutex>
#include <string>
#include <thread>

#include "eval.h"

namespace xmt {

void uci_out(const std::string& s);  // main.cpp (thread-safe)

static inline int score_to_tt(int s, int ply) {
  if (s >= MATE_BOUND) return s + ply;
  if (s <= -MATE_BOUND) return s - ply;
  return s;
}
static inline int score_from_tt(int s, int ply) {
  if (s >= MATE_BOUND) return s - ply;
  if (s <= -MATE_BOUND) return s + ply;
  return s;
}

Worker::Worker(int id_, Searcher* m) : id(id_), mgr(m) {
  for (int d = 0; d < 64; ++d)
    for (int mv = 0; mv < 64; ++mv)
      lmr[d][mv] = (d == 0 || mv == 0) ? 0 : (int)(0.75 + std::log((double)d) * std::log((double)mv) / 2.25);
  nnCache.resize(2048);
  logitBuf.resize(Network::POLICY_SIZE);
  reset();
}

void Worker::reset() {
  std::memset(history, 0, sizeof(history));
  std::memset(killers, 0, sizeof(killers));
  for (auto& e : nnCache) e.key = 0;
  pubNodes = 0;
}

bool Worker::stopped() const { return mgr->stopFlag.load(std::memory_order_relaxed); }

int64_t Worker::elapsed_ms() const {
  return std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - startTime).count();
}

void Worker::check_time() {
  pubNodes.store(nodes, std::memory_order_relaxed);
  if (id != 0 || stopped()) return;
  if (nodeLimit && mgr->total_nodes() >= nodeLimit) { mgr->stopFlag = true; return; }
  if (timed && rootDepth > 1 && elapsed_ms() >= hardMs) mgr->stopFlag = true;
}

// ------------------------------------------------------------ NN (Policy Head priors)
const Worker::NNCacheEntry* Worker::nn_priors(const Move* list, int n) {
  if (!net || !net->loaded() || n > 256 || n == 0) return nullptr;
  NNCacheEntry& e = nnCache[pos.key & (nnCache.size() - 1)];
  if (e.key == pos.key && e.n == n) return &e;
  float value;
  net->evaluate(pos, value, logitBuf.data());
  ++nnEvals;
  float l[256], mx = -1e30f;
  for (int i = 0; i < n; ++i) {
    l[i] = net->move_logit(logitBuf.data(), list[i], pos.stm);
    mx = std::max(mx, l[i]);
  }
  float sum = 0.f;
  for (int i = 0; i < n; ++i) { l[i] = std::exp(l[i] - mx); sum += l[i]; }
  e.key = pos.key;
  e.n = (uint16_t)n;
  e.value = value;
  for (int i = 0; i < n; ++i) e.prob[i] = (uint8_t)std::min(255.f, std::floor(255.f * l[i] / sum + 0.5f));
  return &e;
}

// ------------------------------------------------------------ Move ordering
int Worker::score_move(Move m, Move ttMove, int ply, int prior) const {
  if (m == ttMove) return 1 << 30;
  const int from = from_sq(m), to = to_sq(m), promo = promo_of(m);
  const bool ep = (m & FLAG_EP) != 0;
  const int victimPc = pos.board[to];
  const bool isCap = victimPc != NO_PIECE || ep;
  if (isCap || promo) {
    int victim = ep ? PAWN : (victimPc == NO_PIECE ? PAWN : type_of(victimPc));
    int attacker = type_of(pos.board[from]);
    int mvvlva = SEE_VALUE[victim] * 16 - attacker;
    if (promo) return promo == QUEEN ? 2500000 + mvvlva : -900000 + SEE_VALUE[promo];
    return pos.see_ge(m, 0) ? 2000000 + mvvlva : -1000000 + mvvlva;
  }
  if (m == killers[ply][0]) return 900000;
  if (m == killers[ply][1]) return 800000;
  return history[pos.stm][from][to] + prior * 80;
}

// ------------------------------------------------------------ Quiescence (with SEE)
int Worker::qsearch(int alpha, int beta, int ply) {
  pvLen[ply] = ply;
  ++nodes;
  if ((nodes & 2047) == 0) check_time();
  if (stopped()) return 0;
  if (ply > seldepth) seldepth = ply;
  if (ply > 0 && pos.is_draw(ply)) return 0;
  if (ply >= MAX_PLY - 1) return evaluate(pos);

  const bool inCheck = pos.in_check();
  const uint64_t key = pos.key;
  TTData td;
  Move ttMove = NO_MOVE;
  if (mgr->tt.probe(key, td)) {
    ttMove = td.move;
    int s = score_from_tt(td.score, ply);
    if (td.flag == TT_EXACT || (td.flag == TT_BETA && s >= beta) || (td.flag == TT_ALPHA && s <= alpha)) return s;
  }

  int stand = -INF, best = -INF;
  if (!inCheck) {
    stand = best = evaluate(pos);
    if (stand >= beta) return stand;
    if (stand > alpha) alpha = stand;
  }

  Move list[256];
  int sc[256];
  int n = generate(pos, list, !inCheck);
  for (int i = 0; i < n; ++i) {
    Move m = list[i];
    int victimPc = pos.board[to_sq(m)];
    int victim = (m & FLAG_EP) ? PAWN : (victimPc == NO_PIECE ? -1 : type_of(victimPc));
    sc[i] = (m == ttMove) ? (1 << 30) : (victim >= 0 ? SEE_VALUE[victim] * 16 - type_of(pos.board[from_sq(m)]) : 0) +
                                             (promo_of(m) ? 5000 : 0);
  }

  int moveCount = 0;
  const int origAlpha = alpha;
  Move bestMove = NO_MOVE;
  for (int i = 0; i < n; ++i) {
    int bi = i;
    for (int j = i + 1; j < n; ++j) if (sc[j] > sc[bi]) bi = j;
    std::swap(list[i], list[bi]);
    std::swap(sc[i], sc[bi]);
    Move m = list[i];

    if (!inCheck) {
      int victimPc = pos.board[to_sq(m)];
      int vval = (m & FLAG_EP) ? SEE_VALUE[PAWN] : (victimPc == NO_PIECE ? 0 : SEE_VALUE[type_of(victimPc)]);
      if (!promo_of(m) && stand + vval + 200 <= alpha) continue;  // delta pruning
      if (!pos.see_ge(m, 0)) continue;                             // captura perdedora
    }
    if (!pos.do_move(m)) continue;
    ++moveCount;
    int v = -qsearch(-beta, -alpha, ply + 1);
    pos.undo_move();
    if (stopped()) return 0;
    if (v > best) {
      best = v;
      bestMove = m;
      if (v > alpha) {
        alpha = v;
        if (v >= beta) break;
      }
    }
  }
  if (inCheck && moveCount == 0) return -MATE + ply;

  uint8_t flag = best >= beta ? TT_BETA : (best > origAlpha ? TT_EXACT : TT_ALPHA);
  mgr->tt.store(key, bestMove, score_to_tt(best, ply), 0, flag);
  return best;
}

// ------------------------------------------------------------ Main search (Negamax/PVS)
int Worker::search(int depth, int alpha, int beta, int ply, bool doNull) {
  pvLen[ply] = ply;
  const bool pvNode = (beta - alpha) > 1;
  ++nodes;
  if ((nodes & 2047) == 0) check_time();
  if (stopped()) return 0;
  if (ply > seldepth) seldepth = ply;

  if (pos.is_draw(ply)) return 0;
  if (ply >= MAX_PLY - 2) return evaluate(pos);

  alpha = std::max(alpha, -MATE + ply);
  beta = std::min(beta, MATE - ply - 1);
  if (alpha >= beta) return alpha;

  const bool inCheck = pos.in_check();
  if (inCheck) ++depth;  // check extension
  if (depth <= 0) return qsearch(alpha, beta, ply);

  const uint64_t key = pos.key;
  TTData td;
  Move ttMove = NO_MOVE;
  if (mgr->tt.probe(key, td)) {
    ttMove = td.move;
    if (!pvNode && td.depth >= depth && pos.halfmove < 90) {
      int s = score_from_tt(td.score, ply);
      if (td.flag == TT_EXACT || (td.flag == TT_BETA && s >= beta) || (td.flag == TT_ALPHA && s <= alpha)) return s;
    }
  }

  const int staticEval = inCheck ? 0 : evaluate(pos);
  const int us = pos.stm;

  if (!pvNode && !inCheck) {
    // Reverse futility
    if (depth <= 6 && staticEval - 85 * depth >= beta && beta > -MATE_BOUND && pos.has_non_pawn_material(us))
      return staticEval - 85 * depth;
    // Razoring
    if (depth <= 2 && staticEval + 250 * depth < alpha) {
      int v = qsearch(alpha, alpha + 1, ply);
      if (v <= alpha) return v;
    }
    // Null move
    if (doNull && depth >= 3 && staticEval >= beta && beta > -MATE_BOUND && pos.has_non_pawn_material(us)) {
      int R = 2 + depth / 4 + (staticEval - beta > 200 ? 1 : 0);
      pos.do_null();
      int v = -search(depth - 1 - R, -beta, -beta + 1, ply + 1, false);
      pos.undo_null();
      if (stopped()) return 0;
      if (v >= beta) return v >= MATE_BOUND ? beta : v;
    }
  }

  Move list[256];
  int sc[256];
  int n = generate(pos, list, false);

  const NNCacheEntry* pri = nullptr;
  if (opts.useNN && ply <= opts.nnPly && depth >= 3) pri = nn_priors(list, n);
  for (int i = 0; i < n; ++i) sc[i] = score_move(list[i], ttMove, ply, pri ? pri->prob[i] : 0);

  int bestScore = -INF, moveCount = 0, nq = 0;
  Move bestMove = NO_MOVE, quiets[64];
  const int origAlpha = alpha;

  for (int i = 0; i < n; ++i) {
    int bi = i;
    for (int j = i + 1; j < n; ++j) if (sc[j] > sc[bi]) bi = j;
    std::swap(list[i], list[bi]);
    std::swap(sc[i], sc[bi]);
    const Move m = list[i];

    const int toPc = pos.board[to_sq(m)];
    const bool isCap = toPc != NO_PIECE || (m & FLAG_EP);
    const bool isQuiet = !isCap && !promo_of(m);

    bool pruneIfNoCheck = false;
    if (!pvNode && !inCheck && moveCount > 0 && bestScore > -MATE_BOUND) {
      if (isQuiet) {
        if (depth <= 3 && staticEval + 120 + 130 * depth <= alpha) pruneIfNoCheck = true;
        if (depth <= 4 && moveCount >= 3 + depth * depth) pruneIfNoCheck = true;
      } else if (isCap && depth <= 4 && sc[i] < 0 && !pos.see_ge(m, -100 * depth)) {
        pruneIfNoCheck = true;  // losing exchange at low depth
      }
    }

    if (!pos.do_move(m)) continue;
    const bool givesCheck = pos.in_check();
    if (pruneIfNoCheck && !givesCheck) { pos.undo_move(); continue; }
    ++moveCount;
    if (isQuiet && nq < 64) quiets[nq++] = m;

    const int newDepth = depth - 1;
    int v;
    if (moveCount == 1) {
      v = -search(newDepth, -beta, -alpha, ply + 1, true);
    } else {
      int r = 0;
      if (depth >= 3 && moveCount >= 3 && isQuiet && !inCheck && !givesCheck) {
        r = lmr[std::min(depth, 63)][std::min(moveCount, 63)];
        if (pvNode) --r;
        if (m == killers[ply][0] || m == killers[ply][1]) --r;
        r = std::max(0, std::min(r, newDepth - 1));
      }
      v = -search(newDepth - r, -alpha - 1, -alpha, ply + 1, true);
      if (v > alpha && r > 0) v = -search(newDepth, -alpha - 1, -alpha, ply + 1, true);
      if (v > alpha && v < beta) v = -search(newDepth, -beta, -alpha, ply + 1, true);
    }
    pos.undo_move();
    if (stopped()) return 0;

    if (v > bestScore) {
      bestScore = v;
      bestMove = m;
      if (v > alpha) {
        alpha = v;
        if (pvNode) hash_update_pv(ply, m);
        if (alpha >= beta) {
          if (isQuiet) {
            if (killers[ply][0] != m) { killers[ply][1] = killers[ply][0]; killers[ply][0] = m; }
            int bonus = std::min(depth * depth, 400);
            auto upd = [&](Move mm, int b) {
              int& h = history[us][from_sq(mm)][to_sq(mm)];
              h += b - h * std::abs(b) / 16384;
            };
            upd(m, bonus);
            for (int q = 0; q < nq - 1; ++q) upd(quiets[q], -bonus);
          }
          break;
        }
      }
    }
  }

  if (moveCount == 0) return inCheck ? -MATE + ply : 0;

  uint8_t flag = bestScore >= beta ? TT_BETA : (bestScore > origAlpha ? TT_EXACT : TT_ALPHA);
  mgr->tt.store(key, bestMove, score_to_tt(bestScore, ply), depth, flag);
  return bestScore;
}

void Worker::hash_update_pv(int ply, Move m) {
  pvTable[ply][ply] = m;
  for (int k = ply + 1; k < pvLen[ply + 1]; ++k) pvTable[ply][k] = pvTable[ply + 1][k];
  pvLen[ply] = std::max(ply + 1, pvLen[ply + 1]);
}

// ------------------------------------------------------------ Root
int Worker::search_root(int depth, int alpha, int beta) {
  int best = -INF, fallback = -INF;
  const bool useNet = opts.useNN && net && net->loaded();
  const int margin0 = (useNet && opts.nnMargin > 0) ? opts.nnMargin : 0;
  pvLen[0] = 0;

  for (size_t i = 0; i < root.size(); ++i) {
    RootMove& rm = root[i];
    pos.do_move(rm.move);
    int v;
    bool exact = false;
    if (i == 0) {
      v = -search(depth - 1, -beta, -alpha, 1, true);
      exact = v > alpha && v < beta;
    } else {
      int margin = (best > -INF && std::abs(best) < 600) ? margin0 : 0;
      int lo = std::max(alpha, best > -INF ? best - margin : alpha);
      v = -search(depth - 1, -lo - 1, -lo, 1, true);
      if (v > lo && !stopped()) {
        v = -search(depth - 1, -beta, -lo, 1, true);
        exact = v > lo && v < beta;
      }
    }
    pos.undo_move();
    if (stopped()) return best > -INF ? best : fallback;

    rm.score = v;
    rm.exact = exact;
    if (exact || v >= beta) {
      if (v > best) {
        best = v;
        iterBestMove = rm.move;
        pvTable[0][0] = rm.move;
        for (int k = 1; k < pvLen[1]; ++k) pvTable[0][k] = pvTable[1][k];
        pvLen[0] = std::max(1, pvLen[1]);
      }
      if (v >= beta) break;
    } else {
      fallback = std::max(fallback, v);
    }
  }
  return best > -INF ? best : fallback;
}

void Worker::print_info(int depth, int score, int64_t ms) {
  pubNodes.store(nodes, std::memory_order_relaxed);
  const uint64_t total = mgr->total_nodes();
  std::string s = "info depth " + std::to_string(depth) + " seldepth " + std::to_string(seldepth) + " score ";
  if (score >= MATE_BOUND) s += "mate " + std::to_string((MATE - score + 1) / 2);
  else if (score <= -MATE_BOUND) s += "mate -" + std::to_string((MATE + score) / 2);
  else s += "cp " + std::to_string(score);
  uint64_t nps = ms > 0 ? total * 1000 / (uint64_t)ms : total;
  s += " nodes " + std::to_string(total) + " nps " + std::to_string(nps) + " hashfull " +
       std::to_string(mgr->tt.hashfull()) + " time " + std::to_string(ms) + " pv";
  for (int k = 0; k < pvLen[0]; ++k) s += " " + move_to_uci(pvTable[0][k]);
  uci_out(s);
}

Move Worker::think(const Position& rootPos, const Limits& lim) {
  pos = rootPos;
  startTime = std::chrono::steady_clock::now();
  nodes = 0; nnEvals = 0; seldepth = 0; rootDepth = 0;
  nodeLimit = id == 0 ? lim.nodes : 0;
  pubNodes = 0;
  for (int c = 0; c < 2; ++c)
    for (int a = 0; a < 64; ++a)
      for (int b = 0; b < 64; ++b) history[c][a][b] /= 2;
  std::memset(killers, 0, sizeof(killers));

  const bool useNet = opts.useNN && net && net->loaded();

  // ---- root
  root.clear();
  Move list[256];
  int n = generate(pos, list, false);
  for (int i = 0; i < n; ++i)
    if (pos.do_move(list[i])) { pos.undo_move(); RootMove rm; rm.move = list[i]; root.push_back(rm); }
  if (root.empty()) return NO_MOVE;
  if (root.size() == 1 && !lim.infinite && lim.depth == 0) return root[0].move;
  if (id != 0 && root.size() == 1) return NO_MOVE;

  if (useNet) {
    float value;
    net->evaluate(pos, value, logitBuf.data());
    ++nnEvals;
    float mx = -1e30f;
    std::vector<float> l(root.size());
    for (size_t i = 0; i < root.size(); ++i) { l[i] = net->move_logit(logitBuf.data(), root[i].move, pos.stm); mx = std::max(mx, l[i]); }
    float sum = 0.f;
    for (auto& x : l) { x = std::exp(x - mx); sum += x; }
    for (size_t i = 0; i < root.size(); ++i) root[i].prob = l[i] / sum;
  }
  {
    std::vector<std::pair<double, size_t>> order;
    for (size_t i = 0; i < root.size(); ++i) {
      Move m = root[i].move;
      double k = root[i].prob;
      int victimPc = pos.board[to_sq(m)];
      bool cap = victimPc != NO_PIECE || (m & FLAG_EP);
      if (promo_of(m) == QUEEN) k += 1.5;
      if (cap) k += pos.see_ge(m, 0) ? 1.0 + (victimPc == NO_PIECE ? 0 : SEE_VALUE[type_of(victimPc)]) / 1000.0 : -0.5;
      order.push_back({-k, i});
    }
    std::stable_sort(order.begin(), order.end());
    std::vector<RootMove> sorted;
    for (auto& o : order) sorted.push_back(root[o.second]);
    root = sorted;
  }

  // ---- time management
  timed = false;
  hardMs = softMs = INT64_MAX / 4;
  const int64_t myTime = pos.stm == WHITE ? lim.wtime : lim.btime;
  const int64_t inc = pos.stm == WHITE ? lim.winc : lim.binc;
  if (id == 0 && !lim.infinite && lim.depth == 0 && lim.nodes == 0) {
    if (lim.movetime > 0) {
      timed = true;
      softMs = hardMs = std::max<int64_t>(5, lim.movetime - opts.moveOverhead);
    } else if (myTime >= 0) {
      timed = true;
      double avail = (double)std::max<int64_t>(5, myTime - opts.moveOverhead);
      int mtg = lim.movestogo > 0 ? std::min(lim.movestogo, 40) : 26;
      double soft = std::min(avail / mtg + inc * 0.75, avail * 0.40);
      double hard = std::min(soft * 3.5, avail * 0.60);
      softMs = (int64_t)std::max(5.0, soft);
      hardMs = (int64_t)std::max((double)softMs, hard);
    }
  }

  const int maxDepth = (id == 0 && lim.depth > 0) ? std::min(lim.depth, MAX_PLY - 4) : 64;
  completed = root;
  Move completedBest = root[0].move;
  bool completedValid = false;
  int bestScore = 0, prevScore = 0;

  // Lazy SMP: helpers skip some depths so that threads desynchronize.
  static const int skipSize[20] = {1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4};
  static const int skipPhase[20] = {0, 1, 0, 1, 2, 3, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 6, 7};

  for (int depth = 1; depth <= maxDepth; ++depth) {
    if (id != 0) {
      int k = (id - 1) % 20;
      if (depth > 1 && ((depth + skipPhase[k]) / skipSize[k]) % 2) continue;
    }
    rootDepth = depth;
    iterBestMove = NO_MOVE;
    int alpha = -INF, beta = INF, delta = 0;
    if (depth >= 5) {
      delta = 20;
      alpha = std::max(-INF, prevScore - delta);
      beta = std::min(INF, prevScore + delta);
    }
    int score;
    for (;;) {
      score = search_root(depth, alpha, beta);
      if (stopped()) break;
      if (score <= alpha && alpha > -INF) {
        beta = (alpha + beta) / 2;
        alpha = std::max(-INF, score - delta);
        delta += delta / 2 + 5;
        continue;
      }
      if (score >= beta && beta < INF) {
        beta = std::min(INF, score + delta);
        delta += delta / 2 + 5;
        continue;
      }
      break;
    }
    if (stopped()) break;

    // completed iteration: best move first, rest ordered by score
    for (size_t i = 0; i < root.size(); ++i)
      if (root[i].move == iterBestMove) { std::swap(root[0], root[i]); break; }
    std::stable_sort(root.begin() + 1, root.end(), [](const RootMove& a, const RootMove& b) { return a.score > b.score; });
    completed = root;
    completedBest = iterBestMove;
    completedValid = true;
    bestScore = prevScore = score;
    if (id == 0) print_info(depth, score, elapsed_ms());
    if (id != 0) continue;

    if (!lim.infinite && std::abs(score) >= MATE_BOUND && depth >= (MATE - std::abs(score)) + 2) break;
    if (timed && elapsed_ms() > softMs * 6 / 10) break;
  }

  pubNodes.store(nodes, std::memory_order_relaxed);
  if (id != 0) return completedBest;

  // ---- final move selection
  Move finalMove;
  if (stopped() && iterBestMove != NO_MOVE && iterBestMove != completedBest) {
    finalMove = iterBestMove;  // improvement found in the interrupted iteration
  } else {
    finalMove = completedBest;
    if (completedValid && useNet && opts.nnMargin > 0 && std::abs(bestScore) < 600) {
      std::vector<const RootMove*> cand;
      for (const RootMove& rm : completed)
        if (rm.exact && rm.score >= bestScore - opts.nnMargin) cand.push_back(&rm);
      if (cand.size() > 1) {
        double bestTotal = -1e30;
        for (const RootMove* rm : cand) {
          pos.do_move(rm->move);
          float v;
          net->evaluate(pos, v, nullptr);  // value from the side to move in the child
          pos.undo_move();
          double total = rm->score + opts.nnValueCp * (double)(-v) + opts.nnPolicyCp * (double)rm->prob;
          if (total > bestTotal) { bestTotal = total; finalMove = rm->move; }
        }
        if (finalMove != completedBest)
          uci_out("info string nn tiebreak: " + move_to_uci(completedBest) + " -> " + move_to_uci(finalMove));
      }
    }
  }
  if (finalMove == NO_MOVE) finalMove = root[0].move;
  return finalMove;
}

// ------------------------------------------------------------ Transposition table
// data layout: move(19) | score+32768 (16) | depth (8) | flag (2) | age (8)
static inline uint64_t pack(Move m, int score, int depth, uint8_t flag, uint8_t age) {
  return (uint64_t)(m & 0x7FFFF) | ((uint64_t)(uint16_t)(score + 32768) << 19) |
         ((uint64_t)(uint8_t)std::min(std::max(depth, 0), 255) << 35) | ((uint64_t)(flag & 3) << 43) |
         ((uint64_t)age << 45);
}

void TranspositionTable::resize(size_t mb) {
  size_t bytes = std::max<size_t>(1, mb) * 1024 * 1024;
  size_t n = 1;
  while (n * 2 * sizeof(Entry) <= bytes) n *= 2;
  table.reset(new Entry[n]);
  count = n;
  mask = n - 1;
  clear();
}

void TranspositionTable::clear() {
  for (size_t i = 0; i < count; ++i) {
    table[i].keyx.store(0, std::memory_order_relaxed);
    table[i].data.store(0, std::memory_order_relaxed);
  }
  age = 0;
}

bool TranspositionTable::probe(uint64_t key, TTData& out) const {
  const Entry& e = table[key & mask];
  uint64_t d = e.data.load(std::memory_order_relaxed);
  uint64_t k = e.keyx.load(std::memory_order_relaxed);
  if ((k ^ d) != key) return false;
  out.flag = (uint8_t)((d >> 43) & 3);
  if (out.flag == TT_NONE) return false;
  out.move = (Move)(d & 0x7FFFF);
  out.score = (int)((d >> 19) & 0xFFFF) - 32768;
  out.depth = (int)((d >> 35) & 0xFF);
  return true;
}

void TranspositionTable::store(uint64_t key, Move move, int score, int depth, uint8_t flag) {
  Entry& e = table[key & mask];
  uint64_t d0 = e.data.load(std::memory_order_relaxed);
  uint64_t k0 = e.keyx.load(std::memory_order_relaxed);
  const bool same = (k0 ^ d0) == key;
  const uint8_t flag0 = (uint8_t)((d0 >> 43) & 3);
  const int depth0 = (int)((d0 >> 35) & 0xFF);
  const uint8_t age0 = (uint8_t)((d0 >> 45) & 0xFF);
  if (!same && flag0 != TT_NONE && age0 == age && depth + 2 < depth0) return;
  if (same && move == NO_MOVE) move = (Move)(d0 & 0x7FFFF);
  uint64_t d = pack(move, score, depth, flag, age);
  e.data.store(d, std::memory_order_relaxed);
  e.keyx.store(key ^ d, std::memory_order_relaxed);
}

int TranspositionTable::hashfull() const {
  if (count < 1000) return 0;
  int used = 0;
  for (int i = 0; i < 1000; ++i) {
    uint64_t d = table[i].data.load(std::memory_order_relaxed);
    used += ((d >> 43) & 3) != 0 && (uint8_t)((d >> 45) & 0xFF) == age;
  }
  return used;
}

// ------------------------------------------------------------ Search manager (Lazy SMP)
Searcher::Searcher() {
  tt.resize(64);
  workers.emplace_back(new Worker(0, this));
}

void Searcher::set_hash_mb(int mb) { tt.resize((size_t)mb); }

void Searcher::set_threads(int n) {
  threads = std::max(1, std::min(n, 64));
  while ((int)workers.size() < threads) workers.emplace_back(new Worker((int)workers.size(), this));
  while ((int)workers.size() > threads) workers.pop_back();
}

void Searcher::new_game() {
  tt.clear();
  for (auto& w : workers) w->reset();
}

uint64_t Searcher::total_nodes() const {
  uint64_t n = 0;
  for (const auto& w : workers) n += w->pubNodes.load(std::memory_order_relaxed);
  return n;
}

Move Searcher::think(const Position& rootPos, const Limits& lim) {
  tt.new_search();
  for (auto& w : workers) {
    w->opts = opts;
    w->net = net;
  }
  std::vector<std::thread> helpers;
  for (int i = 1; i < (int)workers.size(); ++i)
    helpers.emplace_back([this, i, &rootPos, &lim]() { workers[i]->think(rootPos, lim); });

  Move m = workers[0]->think(rootPos, lim);
  if (lim.infinite)  // UCI: 'go infinite' must not answer before 'stop'
    while (!stopFlag) std::this_thread::sleep_for(std::chrono::milliseconds(1));
  stopFlag = true;
  for (auto& t : helpers) t.join();
  lastNodes = total_nodes();
  return m;
}

}  // namespace xmt
