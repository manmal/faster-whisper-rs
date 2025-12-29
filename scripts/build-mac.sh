#!/bin/bash
set -e

cd "$(dirname "$0")/.."

# 1. Clone CTranslate2 (The Engine)
if [ ! -d "CTranslate2" ]; then
  git clone --recursive https://github.com/OpenNMT/CTranslate2.git
fi

# 2. Build for macOS (Shared Library)
mkdir -p CTranslate2/build && cd CTranslate2/build
cmake .. \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
  -DWITH_MKL=OFF \
  -DWITH_DNNL=OFF \
  -DWITH_CUDA=OFF \
  -DWITH_RUY=ON \
  -DOPENMP_RUNTIME=COMP \
  -DOpenMP_C_FLAGS="-I/opt/homebrew/opt/libomp/include -Xpreprocessor -fopenmp" \
  -DOpenMP_CXX_FLAGS="-I/opt/homebrew/opt/libomp/include -Xpreprocessor -fopenmp" \
  -DOpenMP_C_LIB_NAMES="omp" \
  -DOpenMP_CXX_LIB_NAMES="omp" \
  -DOpenMP_omp_LIBRARY="/opt/homebrew/opt/libomp/lib/libomp.dylib" \
  -DCMAKE_INSTALL_PREFIX=../../lib_build

make -j$(sysctl -n hw.ncpu)
make install

echo "✅ Built libctranslate2.dylib in lib_build/lib"
