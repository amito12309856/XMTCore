// XMTCore - native UCI engine (C++ search + PolicyValueNet exported from PyTorch)
#include <chrono>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <mutex>
#include <sstream>
#include <thread>

#include "chess.h"
#include "eval.h"
#include "nn.h"
#include "search.h"

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#else
#include <unistd.h>
#endif

using namespace xmt;

#define XMT_VERSION "1.1.0"

static std::mutex g_outMutex;
namespace xmt {
void uci_out(const std::string& s) {
  std::lock_guard<std::mutex> lk(g_outMutex);
  std::fwrite(s.data(), 1, s.size(), stdout);
  std::fputc('\n', stdout);
  std::fflush(stdout);
}
}  // namespace xmt

static std::string exe_dir(const char* argv0) {
#ifdef _WIN32
  char buf[MAX_PATH];
  DWORD n = GetModuleFileNameA(nullptr, buf, MAX_PATH);
  std::string p(buf, n);
#else
  std::string p = argv0 ? argv0 : "";
  char buf[4096];
  ssize_t n = readlink("/proc/self/exe", buf, sizeof(buf) - 1);
  if (n > 0) p.assign(buf, (size_t)n);
#endif
  size_t k = p.find_last_of("/\\");
  return k == std::string::npos ? std::string(".") : p.substr(0, k);
}

// ------------------------------------------------------------------ test utilities
static int run_perft(const std::string& fen, int depth) {
  Position p;
  if (!p.set_fen(fen)) { std::puts("invalid FEN"); return 1; }
  auto t0 = std::chrono::steady_clock::now();
  uint64_t n = perft(p, depth);
  double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  std::printf("perft(%d) = %llu  (%.2fs, %.1f Mnps)\n", depth, (unsigned long long)n, s, n / s / 1e6);
  return 0;
}

static int run_nn_test(const Network& net, const std::string& file) {
  std::ifstream in(file);
  if (!in) { std::printf("cannot open %s\n", file.c_str()); return 1; }
  std::string line;
  int lines = 0, bad = 0;
  double maxV = 0, maxL = 0;
  std::vector<float> logits(Network::POLICY_SIZE);
  while (std::getline(in, line)) {
    if (line.empty() || line[0] == '#') continue;
    // FEN|value|uci:logit,uci:logit,...
    size_t a = line.find('|'), b = line.find('|', a + 1);
    if (a == std::string::npos || b == std::string::npos) continue;
    Position p;
    if (!p.set_fen(line.substr(0, a))) { ++bad; continue; }
    double refV = std::atof(line.substr(a + 1, b - a - 1).c_str());
    float v;
    net.evaluate(p, v, logits.data());
    maxV = std::max(maxV, std::fabs(refV - v));
    std::stringstream ss(line.substr(b + 1));
    std::string tok;
    while (std::getline(ss, tok, ',')) {
      size_t c = tok.find(':');
      if (c == std::string::npos) continue;
      Move m = p.parse_move(tok.substr(0, c));
      double ref = std::atof(tok.substr(c + 1).c_str());
      if (m == NO_MOVE) { ++bad; continue; }
      maxL = std::max(maxL, std::fabs(ref - net.move_logit(logits.data(), m, p.stm)));
    }
    ++lines;
  }
  std::printf("nn-test: %d positions, max|dValue|=%.6f, max|dLogit|=%.6f, errors=%d -> %s\n", lines, maxV, maxL, bad,
              (maxV < 2e-3 && maxL < 2e-2 && !bad && lines) ? "OK" : "FAILED");
  return (maxV < 2e-3 && maxL < 2e-2 && !bad && lines) ? 0 : 2;
}

static int run_bench(Searcher& s, int depth, int threads, int hashMb) {
  s.set_threads(threads);
  s.set_hash_mb(hashMb);
  static const char* fens[] = {
      "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
      "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
      "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
      "r2q1rk1/pP1p2pp/Q4n2/bbp1p3/Np6/1B3NBn/pPPP1PPP/R3K2R b KQ - 0 1",
      "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
      "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10"};
  uint64_t total = 0;
  auto t0 = std::chrono::steady_clock::now();
  for (auto f : fens) {
    Position p;
    p.set_fen(f);
    Limits l;
    l.depth = depth;
    s.new_game();
    s.stopFlag = false;
    Move m = s.think(p, l);
    total += s.last_nodes();
    std::printf("bench %-60s -> %s\n", f, move_to_uci(m).c_str());
  }
  double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  std::printf("bench: %llu nodes in %.2fs = %.0f nps (threads=%d)\n", (unsigned long long)total, sec, total / sec, threads);
  return 0;
}

// ------------------------------------------------------------------ UCI
struct Engine {
  Position pos;
  Searcher searcher;
  Network net;
  std::thread worker;
  std::string nnPath, exeDir;
  int hashMb = 64;

  void load_net() {
    std::string err;
    std::string path = nnPath.empty() ? exeDir + "/xmtcore.nn" : nnPath;
    if (net.load(path, err)) {
      searcher.set_network(&net);
      uci_out("info string network loaded: " + path + " (generation " + std::to_string(net.generation) + ", " +
              std::to_string(net.games) + " games)");
    } else {
      searcher.set_network(nullptr);
      uci_out("info string NO neural network (" + err + ") - using classical evaluation only");
    }
  }

  void stop_search() {
    searcher.stopFlag = true;
    if (worker.joinable()) worker.join();
  }

  void cmd_position(std::istringstream& is) {
    std::string tok;
    is >> tok;
    Position p;
    if (tok == "startpos") {
      p.set_startpos();
      is >> tok;
    } else if (tok == "fen") {
      std::string fen;
      while (is >> tok && tok != "moves") fen += tok + " ";
      if (!p.set_fen(fen)) { uci_out("info string invalid FEN"); return; }
    } else {
      return;
    }
    if (tok == "moves") {
      while (is >> tok) {
        Move m = p.parse_move(tok);
        if (m == NO_MOVE) { uci_out("info string illegal move: " + tok); break; }
        p.do_move(m);
      }
    }
    pos = p;
  }

  void cmd_go(std::istringstream& is) {
    Limits lim;
    std::string t;
    while (is >> t) {
      if (t == "wtime") is >> lim.wtime;
      else if (t == "btime") is >> lim.btime;
      else if (t == "winc") is >> lim.winc;
      else if (t == "binc") is >> lim.binc;
      else if (t == "movetime") is >> lim.movetime;
      else if (t == "movestogo") is >> lim.movestogo;
      else if (t == "depth") is >> lim.depth;
      else if (t == "nodes") is >> lim.nodes;
      else if (t == "infinite") lim.infinite = true;
      else if (t == "ponder") lim.ponder = true;
    }
    stop_search();
    searcher.stopFlag = false;
    Position copy = pos;
    worker = std::thread([this, copy, lim]() {
      Move m = searcher.think(copy, lim);
      Move pm = searcher.last_ponder();
      if (pm != NO_MOVE) uci_out("bestmove " + move_to_uci(m) + " ponder " + move_to_uci(pm));
      else uci_out("bestmove " + move_to_uci(m));
    });
  }

  void cmd_setoption(std::istringstream& is) {
    std::string t, name, value;
    bool inName = false, inValue = false;
    while (is >> t) {
      if (t == "name") { inName = true; inValue = false; continue; }
      if (t == "value") { inValue = true; inName = false; continue; }
      if (inName) name += (name.empty() ? "" : " ") + t;
      else if (inValue) value += (value.empty() ? "" : " ") + t;
    }
    auto num = [&](int lo, int hi, int def) {
      int v = def;
      try { v = std::stoi(value); } catch (...) {}
      return std::max(lo, std::min(hi, v));
    };
    auto boolean = [&]() { return value == "true" || value == "1"; };
    stop_search();
    if (name == "Hash") { hashMb = num(1, 4096, 64); searcher.set_hash_mb(hashMb); }
    else if (name == "Threads") searcher.set_threads(num(1, 64, 1));
    else if (name == "Clear Hash") searcher.new_game();
    else if (name == "Move Overhead") searcher.opts.moveOverhead = num(0, 5000, 100);
    else if (name == "UseNN") searcher.opts.useNN = boolean();
    else if (name == "NNPath") { nnPath = value; load_net(); }
    else if (name == "NNPly") searcher.opts.nnPly = num(0, 6, 2);
    else if (name == "NNMargin") searcher.opts.nnMargin = num(0, 100, 15);
    else if (name == "NNValueCp") searcher.opts.nnValueCp = num(0, 300, 30);
    else if (name == "NNPolicyCp") searcher.opts.nnPolicyCp = num(0, 300, 20);
    // Ponder/SyzygyPath and unknown options: ignored on purpose
  }

  void loop() {
    std::string line;
    while (std::getline(std::cin, line)) {
      while (!line.empty() && (line.back() == '\r' || line.back() == '\n')) line.pop_back();
      std::istringstream is(line);
      std::string cmd;
      if (!(is >> cmd)) continue;
      if (cmd == "uci") {
        uci_out("id name XMTCore " XMT_VERSION);
        uci_out("id author Xamito");
        uci_out("option name Hash type spin default 64 min 1 max 4096");
        uci_out("option name Threads type spin default 1 min 1 max 64");
        uci_out("option name Ponder type check default false");
        uci_out("option name Move Overhead type spin default 100 min 0 max 5000");
        uci_out("option name Clear Hash type button");
        uci_out("option name UseNN type check default true");
        uci_out("option name NNPath type string default <empty>");
        uci_out("option name NNPly type spin default 2 min 0 max 6");
        uci_out("option name NNMargin type spin default 15 min 0 max 100");
        uci_out("option name NNValueCp type spin default 30 min 0 max 300");
        uci_out("option name NNPolicyCp type spin default 20 min 0 max 300");
        uci_out("uciok");
      } else if (cmd == "isready") uci_out("readyok");
      else if (cmd == "ucinewgame") { stop_search(); searcher.new_game(); pos.set_startpos(); }
      else if (cmd == "position") { stop_search(); cmd_position(is); }
      else if (cmd == "go") cmd_go(is);
      else if (cmd == "ponderhit") searcher.ponder_hit();
      else if (cmd == "stop") stop_search();
      else if (cmd == "setoption") cmd_setoption(is);
      else if (cmd == "quit") break;
    }
    stop_search();
  }
};

int main(int argc, char** argv) {
  init_all();
  std::string dir = exe_dir(argc ? argv[0] : nullptr);

  if (argc >= 2 && std::string(argv[1]) == "--version") {
    std::printf("XMTCore %s\n", XMT_VERSION);
    return 0;
  }
  if (argc >= 3 && std::string(argv[1]) == "--perft") {
    std::string fen = argc >= 4 ? argv[3] : "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
    return run_perft(fen, std::atoi(argv[2]));
  }
  if (argc >= 3 && std::string(argv[1]) == "--nn-test") {
    Network net;
    std::string err;
    std::string w = argc >= 4 ? argv[3] : dir + "/xmtcore.nn";
    if (!net.load(w, err)) { std::printf("error: %s\n", err.c_str()); return 1; }
    return run_nn_test(net, argv[2]);
  }

  Engine eng;
  eng.exeDir = dir;
  if (argc >= 2 && std::string(argv[1]) == "--bench") {
    eng.load_net();
    return run_bench(eng.searcher, argc >= 3 ? std::atoi(argv[2]) : 9, argc >= 4 ? std::atoi(argv[3]) : 1,
                     argc >= 5 ? std::atoi(argv[4]) : 64);
  }
  eng.load_net();
  eng.loop();
  return 0;
}
