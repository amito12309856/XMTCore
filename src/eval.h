#pragma once
#include "chess.h"
namespace xmt {
// Static evaluation (centipawns) from the side to move's perspective.
int evaluate(const Position& pos);
}
