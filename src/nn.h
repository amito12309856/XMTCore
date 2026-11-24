// Inferencia nativa da PolicyValueNet (mesma arquitetura do brain.py).
#pragma once
#include <string>
#include <vector>

#include "chess.h"

namespace xmt {

class Network {
 public:
  static constexpr int C = 32;   // canais da trunk
  static constexpr int E = 16;   // embedding da policy
  static constexpr int H = 64;   // oculta da value head
  static constexpr int POLICY_SIZE = 4096;

  bool load(const std::string& path, std::string& err);
  bool loaded() const { return ok_; }

  // value: [-1,1] da perspectiva de quem joga. logits: 4096 floats (ou nullptr).
  void evaluate(const Position& pos, float& value, float* logits) const;

  // Canonical index (from*64+to, mirrored for Black) + promotion bias.
  float move_logit(const float* logits, Move m, int stm) const {
    int f = from_sq(m), t = to_sq(m);
    if (stm == BLACK) { f ^= 56; t ^= 56; }
    return logits[f * 64 + t] + promo_[promo_of(m)];  // promo_of: 0=nenhuma,1=N,2=B,3=R,4=Q
  }

  uint32_t generation = 0, games = 0;

 private:
  bool ok_ = false;
  std::vector<float> w1_, b1_, w2_, b2_, w3_, b3_, pw_, pb_, fw_, tw_, vw1_, vb1_, vw2_;
  float vb2_ = 0.f;
  float promo_[5] = {0, 0, 0, 0, 0};
};

}  // namespace xmt
