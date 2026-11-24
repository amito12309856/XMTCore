#include "syzygy.h"
#include <vector>
extern "C" {
#include "fathom/tbprobe.h"
}

namespace xmt {

void syzygy_init(const std::string& path) {
  if (path.empty() || path == "<empty>") {
    tb_init("");
    return;
  }
  tb_init(path.c_str());
}

int syzygy_max_pieces() {
  return (int)TB_LARGEST;
}

static inline void extract_bbs(const Position& pos, Bitboard& w, Bitboard& b, Bitboard& k, Bitboard& q, Bitboard& r, Bitboard& bi, Bitboard& kn, Bitboard& p) {
  w  = pos.pieces(WHITE, PAWN) | pos.pieces(WHITE, KNIGHT) | pos.pieces(WHITE, BISHOP) | pos.pieces(WHITE, ROOK) | pos.pieces(WHITE, QUEEN) | pos.pieces(WHITE, KING);
  b  = pos.pieces(BLACK, PAWN) | pos.pieces(BLACK, KNIGHT) | pos.pieces(BLACK, BISHOP) | pos.pieces(BLACK, ROOK) | pos.pieces(BLACK, QUEEN) | pos.pieces(BLACK, KING);
  k  = pos.pieces(WHITE, KING)   | pos.pieces(BLACK, KING);
  q  = pos.pieces(WHITE, QUEEN)  | pos.pieces(BLACK, QUEEN);
  r  = pos.pieces(WHITE, ROOK)   | pos.pieces(BLACK, ROOK);
  bi = pos.pieces(WHITE, BISHOP) | pos.pieces(BLACK, BISHOP);
  kn = pos.pieces(WHITE, KNIGHT) | pos.pieces(BLACK, KNIGHT);
  p  = pos.pieces(WHITE, PAWN)   | pos.pieces(BLACK, PAWN);
}

bool syzygy_probe_wdl(const Position& pos, int& score) {
  if (TB_LARGEST == 0) return false;
  if (pos.castling != 0 || pos.halfmove != 0) return false;
  if (popcount(pos.occ) > (int)TB_LARGEST) return false;

  Bitboard w, b, k, q, r, bi, kn, p;
  extract_bbs(pos, w, b, k, q, r, bi, kn, p);
  unsigned ep = (pos.ep > 0 && pos.ep < 64) ? (unsigned)pos.ep : 0u;
  unsigned res = tb_probe_wdl(w, b, k, q, r, bi, kn, p, 0, 0, ep, pos.stm == WHITE);

  if (res == TB_RESULT_FAILED) return false;
  if (res == TB_WIN) score = TB_WIN_SCORE;
  else if (res == TB_LOSS) score = -TB_WIN_SCORE;
  else score = 0;
  return true;
}

bool syzygy_probe_root(Position& pos, Move& bestMove, int& score) {
  if (TB_LARGEST == 0) return false;
  if (pos.castling != 0) return false;
  if (popcount(pos.occ) > (int)TB_LARGEST) return false;

  Bitboard w, b, k, q, r, bi, kn, p;
  extract_bbs(pos, w, b, k, q, r, bi, kn, p);
  unsigned ep = (pos.ep > 0 && pos.ep < 64) ? (unsigned)pos.ep : 0u;
  unsigned results = 0;
  unsigned res = tb_probe_root(w, b, k, q, r, bi, kn, p, (unsigned)pos.halfmove, 0, ep, pos.stm == WHITE, &results);

  if (res == TB_RESULT_FAILED || res == TB_RESULT_CHECKMATE || res == TB_RESULT_STALEMATE)
    return false;

  unsigned wdl = TB_GET_WDL(res);
  int from = (int)TB_GET_FROM(res);
  int to = (int)TB_GET_TO(res);
  unsigned prom = TB_GET_PROMOTES(res);

  int expectedPromo = 0;
  if (prom == TB_PROMOTES_QUEEN) expectedPromo = QUEEN;
  else if (prom == TB_PROMOTES_ROOK) expectedPromo = ROOK;
  else if (prom == TB_PROMOTES_BISHOP) expectedPromo = BISHOP;
  else if (prom == TB_PROMOTES_KNIGHT) expectedPromo = KNIGHT;

  std::vector<Move> moves(256);
  int n = generate(pos, moves.data(), false);
  for (int i = 0; i < n; ++i) {
    Move m = moves.at((size_t)i);
    if (from_sq(m) == from && to_sq(m) == to && promo_of(m) == expectedPromo) {
      if (!pos.do_move(m)) continue;
      pos.undo_move();
      bestMove = m;
      if (wdl == TB_WIN) score = TB_WIN_SCORE;
      else if (wdl == TB_LOSS) score = -TB_WIN_SCORE;
      else score = 0;
      return true;
    }
  }
  return false;
}

}  // namespace xmt
