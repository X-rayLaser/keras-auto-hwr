#!/bin/bash
mkdir -p build
~/emscripten/emsdk/upstream/emscripten/emcc -O1 -std=c++11 argparser.cpp io_utils.cpp token_passing.cpp main.cpp -o build/token_passing.js -s EXPORTED_FUNCTIONS='["_token_passing_js", "_test"]' -s EXTRA_EXPORTED_RUNTIME_METHODS='["ccall", "cwrap"]' --preload-file dictionary -s MODULARIZE=1 -s 'EXPORT_NAME="MyCode"' -s ALLOW_MEMORY_GROWTH=1
