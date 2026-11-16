#!/usr/bin/env python3
"""Generate lt_enum_names.h (cublasLt tile/stage enum -> name) from the local CUDA headers.

Usage: python3 gen_lt_names.py [cublasLt.h path] [output header]
"""
import re
import sys

cublaslt_h = sys.argv[1] if len(sys.argv) > 1 else "/usr/local/cuda/include/cublasLt.h"
out_path = sys.argv[2] if len(sys.argv) > 2 else "lt_enum_names.h"

h = open(cublaslt_h).read()
tiles = re.findall(r"(CUBLASLT_MATMUL_TILE_[0-9A-Zx]+) = (\d+)", h)
stages = re.findall(r"(CUBLASLT_MATMUL_STAGES_[0-9A-Za-z]+) = (\d+)", h)
out = ["#pragma once"]
out.append("static const char* tile_name(int id){ switch(id){")
for n, v in tiles:
    out.append('  case %s: return "%s";' % (v, n[len("CUBLASLT_MATMUL_TILE_"):]))
out.append('  default: return "?"; } }')
out.append("static const char* stages_name(int id){ switch(id){")
for n, v in stages:
    out.append('  case %s: return "%s";' % (v, n[len("CUBLASLT_MATMUL_STAGES_"):]))
out.append('  default: return "?"; } }')
open(out_path, "w").write("\n".join(out) + "\n")
print("generated", len(tiles), "tiles,", len(stages), "stages ->", out_path)
