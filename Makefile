CXX      ?= g++
ARCH     ?= native
CXXFLAGS ?= -O3 -std=c++17 -march=$(ARCH) -flto=auto -DNDEBUG -Wall -Wextra
LDFLAGS  ?= -flto=auto -pthread
SRC       = src/chess.cpp src/eval.cpp src/nn.cpp src/search.cpp src/main.cpp
EXE      ?= XMTCore

$(EXE): $(SRC) src/chess.h src/eval.h src/nn.h src/search.h
	$(CXX) $(CXXFLAGS) -o $@ $(SRC) $(LDFLAGS)

test: $(EXE)
	./$(EXE) --perft 5 | grep -q 4865609 && echo "perft(5) OK"

clean:
	rm -f $(EXE) $(EXE).exe

.PHONY: test clean
