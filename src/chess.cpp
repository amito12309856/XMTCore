#include "chess.h"

#include <sstream>

namespace xmt {

Magic RookMagics[64], BishopMagics[64];
Bitboard KnightAtt[64], KingAtt[64], PawnAtt[2][64];
Bitboard PassedMask[2][64], ShieldMask[2][64];
uint64_t ZobPiece[12][64], ZobCastle[16], ZobEp[8], ZobSide;
int CastleMask[64];
const int SEE_VALUE[7] = {100, 320, 330, 500, 900, 0, 0};

static Bitboard RookTable[102400];
static Bitboard BishopTable[5248];

void init_eval_tables();  // eval.cpp

// ------------------------------------------------------------------ PRNG
struct Rng {
  uint64_t s;
  explicit Rng(uint64_t seed) : s(seed) {}
  uint64_t next() {  // splitmix64
    uint64_t z = (s += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
  }
  uint64_t sparse() { return next() & next() & next(); }
};

// ------------------------------------------------------------------ Bitboards
static Bitboard sliding_attack(int sq, Bitboard occupied, bool rook) {
  static const int rd[4][2] = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};
  static const int bd[4][2] = {{1, 1}, {1, -1}, {-1, 1}, {-1, -1}};
  Bitboard att = 0;
  const int(*d)[2] = rook ? rd : bd;
  for (int i = 0; i < 4; ++i) {
    int f = file_of(sq) + d[i][0], r = rank_of(sq) + d[i][1];
    while (f >= 0 && f < 8 && r >= 0 && r < 8) {
      int s = r * 8 + f;
      att |= sq_bb(s);
      if (occupied & sq_bb(s)) break;
      f += d[i][0];
      r += d[i][1];
    }
  }
  return att;
}

static void init_magics(bool rook, Magic* magics, Bitboard* table) {
  static Bitboard occupancy[4096], reference[4096];
  static int epoch[4096];
  size_t offset = 0;
  for (int sq = 0; sq < 64; ++sq) {
    Bitboard rank_bb = RANK_1 << (8 * rank_of(sq)), file_bb = FILE_A << file_of(sq);
    Bitboard edges = ((RANK_1 | RANK_8) & ~rank_bb) | ((FILE_A | FILE_H) & ~file_bb);
    Magic& m = magics[sq];
    m.mask = sliding_attack(sq, 0, rook) & ~edges;
    m.shift = 64 - popcount(m.mask);
    m.attacks = table + offset;
    int n = 0;
    Bitboard b = 0;
    do {
      occupancy[n] = b;
      reference[n] = sliding_attack(sq, b, rook);
      ++n;
      b = (b - m.mask) & m.mask;
    } while (b);
    offset += (size_t)n;

    Rng rng(0x1234567ULL + sq * 7919ULL + (rook ? 0 : 104729ULL));
    std::memset(epoch, 0, sizeof(epoch));
    int attempt = 0;
    for (;;) {
      Bitboard magic;
      do {
        magic = rng.sparse();
      } while (popcount((m.mask * magic) >> 56) < 6);
      m.magic = magic;
      ++attempt;
      bool ok = true;
      for (int i = 0; i < n; ++i) {
        unsigned idx = (unsigned)((occupancy[i] * magic) >> m.shift);
        if (epoch[idx] < attempt) {
          epoch[idx] = attempt;
          m.attacks[idx] = reference[i];
        } else if (m.attacks[idx] != reference[i]) {
          ok = false;
          break;
        }
      }
      if (ok) break;
    }
  }
}

void init_all() {
  static bool done = false;
  if (done) return;
  done = true;

  init_magics(true, RookMagics, RookTable);
  init_magics(false, BishopMagics, BishopTable);

  for (int s = 0; s < 64; ++s) {
    int f = file_of(s), r = rank_of(s);
    KnightAtt[s] = KingAtt[s] = PawnAtt[0][s] = PawnAtt[1][s] = 0;
    static const int kn[8][2] = {{1, 2}, {2, 1}, {2, -1}, {1, -2}, {-1, -2}, {-2, -1}, {-2, 1}, {-1, 2}};
    for (auto& d : kn) {
      int nf = f + d[0], nr = r + d[1];
      if (nf >= 0 && nf < 8 && nr >= 0 && nr < 8) KnightAtt[s] |= sq_bb(nr * 8 + nf);
    }
    for (int df = -1; df <= 1; ++df)
      for (int dr = -1; dr <= 1; ++dr) {
        if (!df && !dr) continue;
        int nf = f + df, nr = r + dr;
        if (nf >= 0 && nf < 8 && nr >= 0 && nr < 8) KingAtt[s] |= sq_bb(nr * 8 + nf);
      }
    for (int df = -1; df <= 1; df += 2) {
      int nf = f + df;
      if (nf < 0 || nf > 7) continue;
      if (r < 7) PawnAtt[WHITE][s] |= sq_bb((r + 1) * 8 + nf);
      if (r > 0) PawnAtt[BLACK][s] |= sq_bb((r - 1) * 8 + nf);
    }
    // passed-pawn mask: squares ahead on files f-1, f, f+1
    for (int c = 0; c < 2; ++c) {
      Bitboard m = 0, sh = 0;
      for (int rr = 0; rr < 8; ++rr) {
        bool ahead = c == WHITE ? rr > r : rr < r;
        if (!ahead) continue;
        for (int ff = std::max(0, f - 1); ff <= std::min(7, f + 1); ++ff) m |= sq_bb(rr * 8 + ff);
        int dist = std::abs(rr - r);
        if (dist <= 2)
          for (int ff = std::max(0, f - 1); ff <= std::min(7, f + 1); ++ff) sh |= sq_bb(rr * 8 + ff);
      }
      PassedMask[c][s] = m;
      ShieldMask[c][s] = sh;
    }
  }

  Rng rng(0xC0FFEE1234ULL);
  for (int p = 0; p < 12; ++p)
    for (int s = 0; s < 64; ++s) ZobPiece[p][s] = rng.next();
  for (int i = 0; i < 16; ++i) ZobCastle[i] = rng.next();
  for (int i = 0; i < 8; ++i) ZobEp[i] = rng.next();
  ZobSide = rng.next();

  for (int s = 0; s < 64; ++s) CastleMask[s] = 15;
  CastleMask[0] = 15 & ~2;    // a1: perde O-O-O branco
  CastleMask[7] = 15 & ~1;    // h1: perde O-O branco
  CastleMask[4] = 15 & ~3;    // e1
  CastleMask[56] = 15 & ~8;   // a8
  CastleMask[63] = 15 & ~4;   // h8
  CastleMask[60] = 15 & ~12;  // e8

  init_eval_tables();
}

std::string move_to_uci(Move m) {
  if (m == NO_MOVE) return "0000";
  std::string s;
  int f = from_sq(m), t = to_sq(m);
  s += char('a' + file_of(f));
  s += char('1' + rank_of(f));
  s += char('a' + file_of(t));
  s += char('1' + rank_of(t));
  static const char* pc = "pnbrqk";
  if (promo_of(m)) s += pc[promo_of(m)];
  return s;
}

// ------------------------------------------------------------------ Position
void Position::put_piece(int pc, int s) {
  Bitboard b = sq_bb(s);
  int c = color_of(pc), t = type_of(pc);
  board[s] = pc;
  byType[t] |= b;
  byColor[c] |= b;
  occ |= b;
  key ^= ZobPiece[pc][s];
  mg[c] += g_mg[pc][s];
  eg[c] += g_eg[pc][s];
  phase += PHASE_INC[t];
}

void Position::remove_piece(int s) {
  int pc = board[s];
  Bitboard b = sq_bb(s);
  int c = color_of(pc), t = type_of(pc);
  board[s] = NO_PIECE;
  byType[t] &= ~b;
  byColor[c] &= ~b;
  occ &= ~b;
  key ^= ZobPiece[pc][s];
  mg[c] -= g_mg[pc][s];
  eg[c] -= g_eg[pc][s];
  phase -= PHASE_INC[t];
}

void Position::move_piece(int from, int to) {
  int pc = board[from];
  remove_piece(from);
  put_piece(pc, to);
}

void Position::set_startpos() {
  set_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
}

bool Position::set_fen(const std::string& fen) {
  std::istringstream is(fen);
  std::string placement, side, cast, eps;
  int hm = 0, fm = 1;
  if (!(is >> placement >> side >> cast)) return false;
  if (!(is >> eps)) eps = "-";
  if (!(is >> hm)) hm = 0;
  if (!(is >> fm)) fm = 1;

  for (int i = 0; i < 64; ++i) board[i] = NO_PIECE;
  std::memset(byType, 0, sizeof(byType));
  std::memset(byColor, 0, sizeof(byColor));
  occ = 0;
  key = 0;
  mg[0] = mg[1] = eg[0] = eg[1] = 0;
  phase = 0;
  st.clear();

  int r = 7, f = 0;
  for (char ch : placement) {
    if (ch == '/') {
      --r;
      f = 0;
    } else if (ch >= '1' && ch <= '8') {
      f += ch - '0';
    } else {
      const char* pcs = "PNBRQKpnbrqk";
      const char* p = std::strchr(pcs, ch);
      if (!p || r < 0 || f > 7) return false;
      put_piece((int)(p - pcs), r * 8 + f);
      ++f;
    }
  }
  if (popcount(byType[KING] & byColor[WHITE]) != 1 || popcount(byType[KING] & byColor[BLACK]) != 1) return false;

  stm = (side == "b") ? BLACK : WHITE;
  castling = 0;
  for (char ch : cast) {
    if (ch == 'K') castling |= 1;
    if (ch == 'Q') castling |= 2;
    if (ch == 'k') castling |= 4;
    if (ch == 'q') castling |= 8;
  }
  ep = -1;
  if (eps.size() == 2 && eps[0] >= 'a' && eps[0] <= 'h' && eps[1] >= '1' && eps[1] <= '8') {
    int es = (eps[1] - '1') * 8 + (eps[0] - 'a');
    if (PawnAtt[stm ^ 1][es] & pieces(stm, PAWN)) ep = es;
  }
  halfmove = hm;
  fullmove = fm;
  key ^= ZobCastle[castling];
  if (stm == BLACK) key ^= ZobSide;
  if (ep != -1) key ^= ZobEp[file_of(ep)];
  return true;
}

Bitboard Position::attackers_to(int s, Bitboard o) const {
  return (PawnAtt[BLACK][s] & pieces(WHITE, PAWN)) | (PawnAtt[WHITE][s] & pieces(BLACK, PAWN)) |
         (KnightAtt[s] & byType[KNIGHT]) | (KingAtt[s] & byType[KING]) |
         (rook_attacks(s, o) & (byType[ROOK] | byType[QUEEN])) |
         (bishop_attacks(s, o) & (byType[BISHOP] | byType[QUEEN]));
}

bool Position::do_move(Move m) {
  StateInfo s;
  s.key = key;
  s.castling = castling;
  s.ep = ep;
  s.halfmove = halfmove;
  s.captured = NO_PIECE;
  s.move = m;
  s.mg[0] = mg[0]; s.mg[1] = mg[1];
  s.eg[0] = eg[0]; s.eg[1] = eg[1];
  s.phase = phase;

  const int us = stm, them = us ^ 1;
  const int from = from_sq(m), to = to_sq(m), promo = promo_of(m);
  const int pc = board[from];

  if (ep != -1) {
    key ^= ZobEp[file_of(ep)];
    ep = -1;
  }
  ++halfmove;

  if (m & FLAG_EP) {
    int capsq = to + (us == WHITE ? -8 : 8);
    s.captured = board[capsq];
    remove_piece(capsq);
    halfmove = 0;
  } else if (board[to] != NO_PIECE) {
    s.captured = board[to];
    remove_piece(to);
    halfmove = 0;
  }
  move_piece(from, to);

  if (type_of(pc) == PAWN) {
    halfmove = 0;
    if (m & FLAG_DOUBLE) {
      int eps = (from + to) / 2;
      if (PawnAtt[us][eps] & pieces(them, PAWN)) {
        ep = eps;
        key ^= ZobEp[file_of(eps)];
      }
    }
    if (promo) {
      remove_piece(to);
      put_piece(make_piece(us, promo), to);
    }
  }
  if (m & FLAG_CASTLE) {
    if (to == 6) move_piece(7, 5);
    else if (to == 2) move_piece(0, 3);
    else if (to == 62) move_piece(63, 61);
    else if (to == 58) move_piece(56, 59);
  }

  key ^= ZobCastle[castling];
  castling &= CastleMask[from] & CastleMask[to];
  key ^= ZobCastle[castling];

  stm = them;
  key ^= ZobSide;
  if (us == BLACK) ++fullmove;
  st.push_back(s);

  if (attackers_to(king_sq(us), occ) & byColor[them]) {
    undo_move();
    return false;
  }
  return true;
}

void Position::undo_move() {
  const StateInfo s = st.back();
  st.pop_back();
  const Move m = s.move;
  const int us = stm ^ 1;
  stm = us;
  if (us == BLACK) --fullmove;
  const int from = from_sq(m), to = to_sq(m), promo = promo_of(m);

  if (m & FLAG_CASTLE) {
    if (to == 6) move_piece(5, 7);
    else if (to == 2) move_piece(3, 0);
    else if (to == 62) move_piece(61, 63);
    else if (to == 58) move_piece(59, 56);
  }
  if (promo) {
    remove_piece(to);
    put_piece(make_piece(us, PAWN), to);
  }
  move_piece(to, from);
  if (s.captured != NO_PIECE) {
    int capsq = (m & FLAG_EP) ? to + (us == WHITE ? -8 : 8) : to;
    put_piece(s.captured, capsq);
  }
  key = s.key;
  castling = s.castling;
  ep = s.ep;
  halfmove = s.halfmove;
  mg[0] = s.mg[0]; mg[1] = s.mg[1];
  eg[0] = s.eg[0]; eg[1] = s.eg[1];
  phase = s.phase;
}

void Position::do_null() {
  StateInfo s;
  s.key = key; s.castling = castling; s.ep = ep; s.halfmove = halfmove;
  s.captured = NO_PIECE; s.move = NO_MOVE;
  s.mg[0] = mg[0]; s.mg[1] = mg[1]; s.eg[0] = eg[0]; s.eg[1] = eg[1]; s.phase = phase;
  if (ep != -1) {
    key ^= ZobEp[file_of(ep)];
    ep = -1;
  }
  key ^= ZobSide;
  stm ^= 1;
  ++halfmove;
  st.push_back(s);
}

void Position::undo_null() {
  const StateInfo s = st.back();
  st.pop_back();
  stm ^= 1;
  key = s.key; ep = s.ep; halfmove = s.halfmove;
}

bool Position::is_repetition(int ply) const {
  int n = (int)st.size();
  int lim = std::min(halfmove, n);
  int cnt = 0;
  for (int i = 4; i <= lim; i += 2) {
    if (st[n - i].key == key) {
      if (i <= ply) return true;  // repetition inside the search tree
      if (++cnt >= 2) return true;  // terceira ocorrencia na partida
    }
  }
  return false;
}

bool Position::insufficient_material() const {
  if (byType[PAWN] | byType[ROOK] | byType[QUEEN]) return false;
  int minors = popcount(byType[KNIGHT] | byType[BISHOP]);
  return minors <= 1;
}

static inline bool aligned_ray(int ksq, int s1, int s2) {
  Bitboard b1 = sq_bb(s1), b2 = sq_bb(s2);
  if ((rook_attacks(ksq, b1) & b2) || (rook_attacks(ksq, b2) & b1)) {
    return (rook_attacks(ksq, 0) & b1) && (rook_attacks(ksq, 0) & b2) &&
           ((rook_attacks(ksq, b1) & rook_attacks(s1, sq_bb(ksq)) & b2) ||
            (rook_attacks(ksq, b2) & rook_attacks(s2, sq_bb(ksq)) & b1));
  }
  if ((bishop_attacks(ksq, b1) & b2) || (bishop_attacks(ksq, b2) & b1)) {
    return (bishop_attacks(ksq, 0) & b1) && (bishop_attacks(ksq, 0) & b2) &&
           ((bishop_attacks(ksq, b1) & bishop_attacks(s1, sq_bb(ksq)) & b2) ||
            (bishop_attacks(ksq, b2) & bishop_attacks(s2, sq_bb(ksq)) & b1));
  }
  return false;
}

// Pin-aware Static Exchange Evaluation (SEE).
bool Position::see_ge(Move m, int threshold) const {
  if (m & (FLAG_CASTLE | FLAG_EP)) return 0 >= threshold;
  const int from = from_sq(m), to = to_sq(m), promo = promo_of(m);
  int swap = (*(board + to) == NO_PIECE ? 0 : *(SEE_VALUE + type_of(*(board + to)))) - threshold;
  if (promo) swap += *(SEE_VALUE + promo) - *(SEE_VALUE + PAWN);
  if (swap < 0) return false;
  swap = (promo ? *(SEE_VALUE + promo) : *(SEE_VALUE + type_of(*(board + from)))) - swap;
  if (swap <= 0) return true;

  Bitboard o = occ ^ sq_bb(from) ^ sq_bb(to);
  int side = color_of(*(board + from));
  Bitboard attackers = attackers_to(to, o) & o;
  int res = 1;
  const Bitboard diag = *(byType + BISHOP) | *(byType + QUEEN);
  const Bitboard orth = *(byType + ROOK) | *(byType + QUEEN);

  int ksqW = king_sq(WHITE), ksqB = king_sq(BLACK);
  Bitboard pinnedW = 0, pinnedB = 0, pinnerW = 0, pinnerB = 0;
  for (int c = 0; c < 2; ++c) {
    int ksq = (c == WHITE) ? ksqW : ksqB;
    Bitboard snipers = ((rook_attacks(ksq, 0) & orth) | (bishop_attacks(ksq, 0) & diag)) & *(byColor + (c ^ 1));
    while (snipers) {
      int s = pop_lsb(snipers);
      Bitboard between = (rook_attacks(ksq, sq_bb(s)) & rook_attacks(s, sq_bb(ksq))) |
                         (bishop_attacks(ksq, sq_bb(s)) & bishop_attacks(s, sq_bb(ksq)));
      Bitboard blk = between & occ;
      if (blk && !more_than_one(blk) && (blk & *(byColor + c))) {
        if (c == WHITE) { pinnedW |= blk; pinnerW |= sq_bb(s); }
        else { pinnedB |= blk; pinnerB |= sq_bb(s); }
      }
    }
  }

  for (;;) {
    side ^= 1;
    attackers &= o;
    Bitboard mine = attackers & *(byColor + side);
    Bitboard sidePinners = (side == WHITE) ? pinnerW : pinnerB;
    if (sidePinners & o) {
      Bitboard sidePinned = (side == WHITE) ? pinnedW : pinnedB;
      int sideKsq = (side == WHITE) ? ksqW : ksqB;
      Bitboard p = mine & sidePinned;
      while (p) {
        int psq = pop_lsb(p);
        if (!aligned_ray(sideKsq, psq, to)) mine ^= sq_bb(psq);
      }
    }
    if (!mine) break;
    res ^= 1;
    Bitboard bb;
    if ((bb = mine & *(byType + PAWN))) {
      if ((swap = *(SEE_VALUE + PAWN) - swap) < res) break;
      o ^= bb & (~bb + 1);
      attackers |= bishop_attacks(to, o) & diag;
    } else if ((bb = mine & *(byType + KNIGHT))) {
      if ((swap = *(SEE_VALUE + KNIGHT) - swap) < res) break;
      o ^= bb & (~bb + 1);
    } else if ((bb = mine & *(byType + BISHOP))) {
      if ((swap = *(SEE_VALUE + BISHOP) - swap) < res) break;
      o ^= bb & (~bb + 1);
      attackers |= bishop_attacks(to, o) & diag;
    } else if ((bb = mine & *(byType + ROOK))) {
      if ((swap = *(SEE_VALUE + ROOK) - swap) < res) break;
      o ^= bb & (~bb + 1);
      attackers |= rook_attacks(to, o) & orth;
    } else if ((bb = mine & *(byType + QUEEN))) {
      if ((swap = *(SEE_VALUE + QUEEN) - swap) < res) break;
      o ^= bb & (~bb + 1);
      attackers |= (bishop_attacks(to, o) & diag) | (rook_attacks(to, o) & orth);
    } else {
      return (attackers & ~(*(byColor + side))) ? (res ^ 1) : res;
    }
  }
  return res != 0;
}

Move Position::parse_move(const std::string& s) {
  if (s.size() < 4) return NO_MOVE;
  int f = (s[0] - 'a') + (s[1] - '1') * 8, t = (s[2] - 'a') + (s[3] - '1') * 8;
  int promo = 0;
  if (s.size() >= 5) {
    switch (s[4]) {
      case 'n': promo = KNIGHT; break;
      case 'b': promo = BISHOP; break;
      case 'r': promo = ROOK; break;
      case 'q': promo = QUEEN; break;
      default: return NO_MOVE;
    }
  }
  Move list[256];
  int n = generate(*this, list, false);
  for (int i = 0; i < n; ++i) {
    Move m = list[i];
    if (from_sq(m) == f && to_sq(m) == t && promo_of(m) == promo) {
      if (do_move(m)) {
        undo_move();
        return m;
      }
    }
  }
  return NO_MOVE;
}

// ------------------------------------------------------------------ Movegen
template <int Us>
static Move* gen_pawns(const Position& pos, Move* m, bool capsOnly) {
  constexpr int Up = Us == WHITE ? 8 : -8;
  constexpr Bitboard Rank7 = Us == WHITE ? RANK_7 : RANK_2;
  constexpr Bitboard Rank3 = Us == WHITE ? RANK_3 : RANK_6;
  const Bitboard occ = pos.occ, theirs = pos.byColor[Us ^ 1];
  const Bitboard pawns = pos.pieces(Us, PAWN);
  auto sh = [](Bitboard b, int d) { return d > 0 ? b << d : b >> (-d); };

  const Bitboard pPromo = pawns & Rank7, pOther = pawns & ~Rank7;

  if (!capsOnly) {
    Bitboard p1 = sh(pOther, Up) & ~occ;
    Bitboard p2 = sh(p1 & Rank3, Up) & ~occ;
    while (p1) { int to = pop_lsb(p1); *m++ = make_move(to - Up, to); }
    while (p2) { int to = pop_lsb(p2); *m++ = make_move(to - 2 * Up, to, 0, FLAG_DOUBLE); }
  }
  // promocoes por avanco
  Bitboard pp = sh(pPromo, Up) & ~occ;
  while (pp) {
    int to = pop_lsb(pp), from = to - Up;
    *m++ = make_move(from, to, QUEEN);
    if (!capsOnly) {
      *m++ = make_move(from, to, ROOK);
      *m++ = make_move(from, to, BISHOP);
      *m++ = make_move(from, to, KNIGHT);
    }
  }
  // captures (left: Up-1 excluding file A; right: Up+1 excluding file H)
  for (int side = 0; side < 2; ++side) {
    const int d = side == 0 ? Up - 1 : Up + 1;
    const Bitboard mask = side == 0 ? ~FILE_A : ~FILE_H;
    Bitboard caps = sh(pOther & mask, d) & theirs;
    while (caps) { int to = pop_lsb(caps); *m++ = make_move(to - d, to); }
    Bitboard pcaps = sh(pPromo & mask, d) & theirs;
    while (pcaps) {
      int to = pop_lsb(pcaps), from = to - d;
      *m++ = make_move(from, to, QUEEN);
      if (!capsOnly) {
        *m++ = make_move(from, to, ROOK);
        *m++ = make_move(from, to, BISHOP);
        *m++ = make_move(from, to, KNIGHT);
      }
    }
  }
  if (pos.ep != -1) {
    Bitboard a = PawnAtt[Us ^ 1][pos.ep] & pawns;
    while (a) *m++ = make_move(pop_lsb(a), pos.ep, 0, FLAG_EP);
  }
  return m;
}

int generate(const Position& pos, Move* out, bool capsOnly) {
  Move* m = out;
  const int us = pos.stm, them = us ^ 1;
  const Bitboard occ = pos.occ, ours = pos.byColor[us], theirs = pos.byColor[them];
  const Bitboard target = capsOnly ? theirs : ~ours;

  m = us == WHITE ? gen_pawns<WHITE>(pos, m, capsOnly) : gen_pawns<BLACK>(pos, m, capsOnly);

  Bitboard b = pos.pieces(us, KNIGHT);
  while (b) {
    int f = pop_lsb(b);
    Bitboard a = KnightAtt[f] & target;
    while (a) *m++ = make_move(f, pop_lsb(a));
  }
  b = pos.pieces(us, BISHOP);
  while (b) {
    int f = pop_lsb(b);
    Bitboard a = bishop_attacks(f, occ) & target;
    while (a) *m++ = make_move(f, pop_lsb(a));
  }
  b = pos.pieces(us, ROOK);
  while (b) {
    int f = pop_lsb(b);
    Bitboard a = rook_attacks(f, occ) & target;
    while (a) *m++ = make_move(f, pop_lsb(a));
  }
  b = pos.pieces(us, QUEEN);
  while (b) {
    int f = pop_lsb(b);
    Bitboard a = queen_attacks(f, occ) & target;
    while (a) *m++ = make_move(f, pop_lsb(a));
  }
  {
    int f = pos.king_sq(us);
    Bitboard a = KingAtt[f] & target;
    while (a) *m++ = make_move(f, pop_lsb(a));

    if (!capsOnly && pos.castling) {
      auto att = [&](int s) { return (pos.attackers_to(s, occ) & theirs) != 0; };
      if (us == WHITE && f == 4) {
        if ((pos.castling & 1) && !(occ & (sq_bb(5) | sq_bb(6))) && !att(4) && !att(5) && !att(6))
          *m++ = make_move(4, 6, 0, FLAG_CASTLE);
        if ((pos.castling & 2) && !(occ & (sq_bb(1) | sq_bb(2) | sq_bb(3))) && !att(4) && !att(3) && !att(2))
          *m++ = make_move(4, 2, 0, FLAG_CASTLE);
      } else if (us == BLACK && f == 60) {
        if ((pos.castling & 4) && !(occ & (sq_bb(61) | sq_bb(62))) && !att(60) && !att(61) && !att(62))
          *m++ = make_move(60, 62, 0, FLAG_CASTLE);
        if ((pos.castling & 8) && !(occ & (sq_bb(57) | sq_bb(58) | sq_bb(59))) && !att(60) && !att(59) && !att(58))
          *m++ = make_move(60, 58, 0, FLAG_CASTLE);
      }
    }
  }
  return (int)(m - out);
}

uint64_t perft(Position& pos, int depth) {
  if (depth == 0) return 1;
  Move list[256];
  int n = generate(pos, list, false);
  uint64_t count = 0;
  for (int i = 0; i < n; ++i) {
    if (!pos.do_move(list[i])) continue;
    count += (depth == 1) ? 1 : perft(pos, depth - 1);
    pos.undo_move();
  }
  return count;
}

}  // namespace xmt
