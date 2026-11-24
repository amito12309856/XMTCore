#include "nn.h"

#include <cmath>
#include <cstdio>

namespace xmt {

static bool read_floats(FILE* f, std::vector<float>& v, size_t n) {
  v.resize(n);
  return fread(v.data(), sizeof(float), n, f) == n;
}

bool Network::load(const std::string& path, std::string& err) {
  ok_ = false;
  FILE* f = fopen(path.c_str(), "rb");
  if (!f) { err = "cannot open " + path; return false; }
  uint32_t hdr[8];
  char magic[4];
  if (fread(magic, 1, 4, f) != 4 || std::memcmp(magic, "XMTN", 4) != 0 || fread(hdr, 4, 8, f) != 8) {
    fclose(f); err = "invalid header (expected XMTN)"; return false;
  }
  // hdr: version, in_channels, channels, embedding, value_hidden, generation, games, reserved
  if (hdr[0] != 1 || hdr[1] != 12 || hdr[2] != C || hdr[3] != E || hdr[4] != H) {
    fclose(f); err = "architecture differs from the compiled one (12/32/16/64)"; return false;
  }
  generation = hdr[5];
  games = hdr[6];
  bool good = read_floats(f, w1_, 9 * 12 * C) && read_floats(f, b1_, C) && read_floats(f, w2_, 9 * C * C) &&
              read_floats(f, b2_, C) && read_floats(f, w3_, 9 * C * C) && read_floats(f, b3_, C) &&
              read_floats(f, pw_, E * C) && read_floats(f, pb_, E) && read_floats(f, fw_, E * E) &&
              read_floats(f, tw_, E * E) && read_floats(f, vw1_, H * C) && read_floats(f, vb1_, H) &&
              read_floats(f, vw2_, H);
  std::vector<float> tmp;
  good = good && fread(promo_, sizeof(float), 5, f) == 5 && fread(&vb2_, sizeof(float), 1, f) == 1;
  fclose(f);
  if (!good) { err = "truncated file"; return false; }
  ok_ = true;
  return true;
}

// HWC layout with a zero border (10x10xC): vectorizes along the channel axis.
static inline void conv3x3(const float* in, float* out, const float* w, const float* b, int C) {
  float acc[Network::C];
  for (int oy = 0; oy < 8; ++oy)
    for (int ox = 0; ox < 8; ++ox) {
      for (int c = 0; c < C; ++c) acc[c] = b[c];
      for (int ky = 0; ky < 3; ++ky)
        for (int kx = 0; kx < 3; ++kx) {
          const float* p = in + ((oy + ky) * 10 + (ox + kx)) * C;
          const float* wk = w + (ky * 3 + kx) * C * C;
          for (int ci = 0; ci < C; ++ci) {
            float a = p[ci];
            if (a == 0.f) continue;  // post-ReLU: about half of the values are zero
            const float* wr = wk + ci * C;
            for (int co = 0; co < Network::C; ++co) acc[co] += a * wr[co];
          }
        }
      float* o = out + ((oy + 1) * 10 + (ox + 1)) * C;
      for (int c = 0; c < C; ++c) o[c] = acc[c] > 0.f ? acc[c] : 0.f;
    }
}

void Network::evaluate(const Position& pos, float& value, float* logits) const {
  alignas(32) float A[100 * C];
  alignas(32) float B[100 * C];
  std::memset(A, 0, sizeof(A));
  std::memset(B, 0, sizeof(B));

  // Layer 1 (sparse binary input: sum the kernel of each piece)
  for (int y = 0; y < 8; ++y)
    for (int x = 0; x < 8; ++x) {
      float* o = A + ((y + 1) * 10 + (x + 1)) * C;
      for (int c = 0; c < C; ++c) o[c] = b1_[c];
    }
  const int stm = pos.stm;
  Bitboard occ = pos.occ;
  while (occ) {
    int sq = pop_lsb(occ);
    int pc = pos.board[sq];
    int ch = type_of(pc) + (color_of(pc) != stm ? 6 : 0);
    int rel = stm == WHITE ? sq : (sq ^ 56);
    int y = 7 - (rel >> 3), x = rel & 7;
    for (int ky = 0; ky < 3; ++ky)
      for (int kx = 0; kx < 3; ++kx) {
        int oy = y - ky + 1, ox = x - kx + 1;
        if (oy < 0 || oy > 7 || ox < 0 || ox > 7) continue;
        float* o = A + ((oy + 1) * 10 + (ox + 1)) * C;
        const float* w = w1_.data() + ((ky * 3 + kx) * 12 + ch) * C;
        for (int c = 0; c < C; ++c) o[c] += w[c];
      }
  }
  for (int y = 0; y < 8; ++y)
    for (int x = 0; x < 8; ++x) {
      float* o = A + ((y + 1) * 10 + (x + 1)) * C;
      for (int c = 0; c < C; ++c) o[c] = o[c] > 0.f ? o[c] : 0.f;
    }

  conv3x3(A, B, w2_.data(), b2_.data(), C);
  conv3x3(B, A, w3_.data(), b3_.data(), C);  // features finais em A

  // Value head
  float mean[C];
  for (int c = 0; c < C; ++c) mean[c] = 0.f;
  for (int y = 0; y < 8; ++y)
    for (int x = 0; x < 8; ++x) {
      const float* p = A + ((y + 1) * 10 + (x + 1)) * C;
      for (int c = 0; c < C; ++c) mean[c] += p[c];
    }
  for (int c = 0; c < C; ++c) mean[c] *= (1.f / 64.f);
  float out = vb2_;
  for (int j = 0; j < H; ++j) {
    float h = vb1_[j];
    const float* wr = vw1_.data() + j * C;
    for (int c = 0; c < C; ++c) h += wr[c] * mean[c];
    if (h > 0.f) out += vw2_[j] * h;
  }
  value = std::tanh(out);

  if (!logits) return;

  // Policy head: src[i] . dst[j] / sqrt(E), i,j = row-major position on the 8x8 grid (row 0 = rank 8)
  float src[64][E], dst[64][E];
  for (int p = 0; p < 64; ++p) {
    const float* f = A + (((p >> 3) + 1) * 10 + ((p & 7) + 1)) * C;
    float sq[E];
    for (int e = 0; e < E; ++e) {
      float s = pb_[e];
      const float* wr = pw_.data() + e * C;
      for (int c = 0; c < C; ++c) s += wr[c] * f[c];
      sq[e] = s;
    }
    for (int k = 0; k < E; ++k) {
      float a = 0.f, b = 0.f;
      const float* fr = fw_.data() + k * E;
      const float* tr = tw_.data() + k * E;
      for (int e = 0; e < E; ++e) { a += fr[e] * sq[e]; b += tr[e] * sq[e]; }
      src[p][k] = a;
      dst[p][k] = b;
    }
  }
  for (int i = 0; i < 64; ++i)
    for (int j = 0; j < 64; ++j) {
      float s = 0.f;
      for (int k = 0; k < E; ++k) s += src[i][k] * dst[j][k];
      logits[i * 64 + j] = s * 0.25f;
    }
}

}  // namespace xmt
