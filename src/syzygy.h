#pragma once
#include <string>
#include "chess.h"

namespace xmt {

constexpr int TB_WIN_SCORE = 20000;

void syzygy_init(const std::string& path);
int syzygy_max_pieces();
bool syzygy_probe_wdl(const Position& pos, int& score);
bool syzygy_probe_root(Position& pos, Move& bestMove, int& score);

}  // namespace xmt
