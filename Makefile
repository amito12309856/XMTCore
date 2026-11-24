CC       ?= gcc
CXX      ?= g++
ARCH     ?= native
CFLAGS   ?= -O3 -DNDEBUG -Isrc/fathom
CXXFLAGS ?= -O3 -std=c++17 -march=$(ARCH) -flto=auto -DNDEBUG -Wall -Wextra -Isrc/fathom
LDFLAGS  ?= -flto=auto -pthread
SRC       = src/chess.cpp src/eval.cpp src/nn.cpp src/search.cpp src/syzygy.cpp src/main.cpp
EXE      ?= XMTCore

$(EXE): $(SRC) src/fathom/tbprobe.c src/chess.h src/eval.h src/nn.h src/search.h src/syzygy.h
$(CC) $(CFLAGS) -c src/fathom/tbprobe.c -o tbprobe.o
$(CXX) $(CXXFLAGS) -o $@ $(SRC) tbprobe.o $(LDFLAGS)

test: $(EXE)
./$(EXE) --perft 5 | grep -q 4865609 && echo "perft(5) OK"

clean:
rm -f $(EXE) $(EXE).exe tbprobe.o

.PHONY: test clean
