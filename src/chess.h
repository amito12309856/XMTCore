// XMTCore - base types, bitboards, position, move generation, SEE.
#pragma once
#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#if defined(_MSC_VER)
#include <intrin.h>
#endif

namespace xmt {

typedef uint64_t Bitboard;
typedef uint32_t Move;
constexpr Move NO_MOVE = 0;

enum Color { WHITE = 0, BLACK = 1 };
enum PieceType { PAWN = 0, KNIGHT = 1, BISHOP = 2, ROOK = 3, QUEEN = 4, KING = 5, NO_PT = 6 };
constexpr int NO_PIECE = 12;  // piece = color*6 + type

// Squares use the python-chess numbering: a1=0 ... h1=7 ... a8=56 ... h8=63
constexpr Bitboard FILE_A = 0x0101010101010101ULL;
constexpr Bitboard FILE_H = FILE_A << 7;
constexpr Bitboard RANK_1 = 0xFFULL;
constexpr Bitboard RANK_2 = RANK_1 << 8;
constexpr Bitboard RANK_3 = RANK_1 << 16;
constexpr Bitboard RANK_6 = RANK_1 << 40;
constexpr Bitboard RANK_7 = RANK_1 << 48;
constexpr Bitboard RANK_8 = RANK_1 << 56;

inline int make_piece(int c, int pt) { return c * 6 + pt; }
inline int color_of(int p) { return p / 6; }
inline int type_of(int p) { return p % 6; }
inline Bitboard sq_bb(int s) { return 1ULL << s; }
inline int file_of(int s) { return s & 7; }
inline int rank_of(int s) { return s >> 3; }

inline int popcount(Bitboard b) {
#if defined(_MSC_VER)
  return (int)__popcnt64(b);
#else
  return __builtin_popcountll(b);
#endif
}
inline int lsb(Bitboard b) {
#if defined(_MSC_VER)
  unsigned long i;
  _BitScanForward64(&i, b);
  return (int)i;
#else
  return __builtin_ctzll(b);
#endif
}
inline int pop_lsb(Bitboard& b) {
  int s = lsb(b);
  b &= b - 1;
  return s;
}
inline bool more_than_one(Bitboard b) { return b & (b - 1); }

// ---- Moves (32 bits): from(6) to(6) promo(3: piece type) flags
constexpr int FLAG_EP = 1 << 16;
constexpr int FLAG_CASTLE = 1 << 17;
constexpr int FLAG_DOUBLE = 1 << 18;
inline Move make_move(int from, int to, int promo = 0, int flags = 0) {
  return (Move)(from | (to << 6) | (promo << 12) | flags);
}
inline int from_sq(Move m) { return m & 63; }
inline int to_sq(Move m) { return (m >> 6) & 63; }
inline int promo_of(Move m) { return (m >> 12) & 7; }
inline uint16_t move16(Move m) { return (uint16_t)(m & 0xFFFF); }
std::string move_to_uci(Move m);

// ---- Tabelas globais
struct Magic {
  Bitboard mask;
  Bitboard magic;
  Bitboard* attacks;
  unsigned shift;
};
extern Magic RookMagics[64], BishopMagics[64];
extern Bitboard KnightAtt[64], KingAtt[64], PawnAtt[2][64];
extern Bitboard PassedMask[2][64], ShieldMask[2][64];
extern uint64_t ZobPiece[12][64], ZobCastle[16], ZobEp[8], ZobSide;
extern int CastleMask[64];
extern int g_mg[12][64], g_eg[12][64];  // value + PST per piece/square (eval.cpp)
extern const int PHASE_INC[6];
extern const int SEE_VALUE[7];

inline Bitboard rook_attacks(int s, Bitboard occ) {
  const Magic& m = RookMagics[s];
  return m.attacks[((occ & m.mask) * m.magic) >> m.shift];
}
inline Bitboard bishop_attacks(int s, Bitboard occ) {
  const Magic& m = BishopMagics[s];
  return m.attacks[((occ & m.mask) * m.magic) >> m.shift];
}
inline Bitboard queen_attacks(int s, Bitboard occ) { return rook_attacks(s, occ) | bishop_attacks(s, occ); }

void init_all();  // call once at startup

struct StateInfo {
  uint64_t key;
  int castling, ep, halfmove;
  int captured;
  Move move;
  int mg[2], eg[2], phase;
};

class Position {
 public:
  int board[64];
  Bitboard byType[6], byColor[2], occ;
  int stm, castling, ep, halfmove, fullmove;
  uint64_t key;
  int mg[2], eg[2], phase;
  std::vector<StateInfo> st;

  Position() { st.reserve(1024); set_startpos(); }
  void set_startpos();
  bool set_fen(const std::string& fen);

  Bitboard pieces(int c, int pt) const { return byColor[c] & byType[pt]; }
  int king_sq(int c) const { return lsb(byColor[c] & byType[KING]); }
  Bitboard attackers_to(int s, Bitboard o) const;
  bool in_check() const { return attackers_to(king_sq(stm), occ) & byColor[stm ^ 1]; }
  bool has_non_pawn_material(int c) const { return byColor[c] & ~(byType[PAWN] | byType[KING]); }

  bool do_move(Move m);  // returns false (and undoes) if the own king is left in check
  void undo_move();
  void do_null();
  void undo_null();

  bool see_ge(Move m, int threshold) const;
  bool is_repetition(int ply) const;
  bool insufficient_material() const;
  bool is_draw(int ply) const { return halfmove >= 100 || insufficient_material() || is_repetition(ply); }
  Move parse_move(const std::string& s);  // NO_MOVE se invalido/ilegal

 private:
  void put_piece(int pc, int s);
  void remove_piece(int s);
  void move_piece(int from, int to);
};

// Generates pseudo-legal moves. capsOnly: captures, en passant and queen promotions.
int generate(const Position& pos, Move* out, bool capsOnly);
uint64_t perft(Position& pos, int depth);

}  // namespace xmt
