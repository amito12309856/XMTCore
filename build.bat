@echo off
REM Build XMTCore with MSVC. Run from "x64 Native Tools Command Prompt for VS 2022".
REM   build.bat            AVX2 build  -> XMTCore.exe
REM   build.bat generic    baseline x64 build (no AVX2) -> XMTCore-generic.exe
set ARCHFLAG=/arch:AVX2
set OUT=XMTCore.exe
if /I "%1"=="generic" (
  set ARCHFLAG=
  set OUT=XMTCore-generic.exe
)
cl /nologo /O2 /GL /std:c++17 /EHsc %ARCHFLAG% /DNDEBUG /Fe:%OUT% src\chess.cpp src\eval.cpp src\nn.cpp src\search.cpp src\main.cpp
if errorlevel 1 (
  echo BUILD FAILED
  exit /b 1
)
echo Built: %OUT%
echo Test:  %OUT% --perft 5   (expected 4865609)
