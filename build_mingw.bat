@echo off
REM Build XMTCore with MSYS2 / MinGW-w64 (g++, posix threads).
REM   build_mingw.bat            x86-64-v3 (AVX2) build
REM   build_mingw.bat generic    baseline x86-64 build
set ARCH=x86-64-v3
set OUT=XMTCore.exe
if /I "%1"=="generic" (
  set ARCH=x86-64
  set OUT=XMTCore-generic.exe
)
g++ -O3 -std=c++17 -march=%ARCH% -flto -DNDEBUG -static -o %OUT% src/chess.cpp src/eval.cpp src/nn.cpp src/search.cpp src/main.cpp -pthread
if errorlevel 1 (
  echo BUILD FAILED
  exit /b 1
)
echo Built: %OUT%
