@echo off
gcc -O3 -DNDEBUG -Isrc/fathom -c src/fathom/tbprobe.c -o tbprobe.o
g++ -O3 -std=c++17 -march=native -mtune=native -flto -DNDEBUG -static -Isrc/fathom -o XMTCore.exe src/chess.cpp src/eval.cpp src/nn.cpp src/search.cpp src/syzygy.cpp src/main.cpp tbprobe.o -pthread
del tbprobe.o
