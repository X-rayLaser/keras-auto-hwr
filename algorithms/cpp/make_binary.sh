#!/bin/bash
mkdir -p build
g++ -o build/token_passing argparser.cpp io_utils.cpp token_passing.cpp main.cpp

