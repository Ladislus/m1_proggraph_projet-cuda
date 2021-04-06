if [ ! -d "cmake-build-debug" ]; then
  mkdir "cmake-build-debug"
fi
if [ ! -d "output" ]; then
  mkdir "cmake-build-debug"
fi
cd "cmake-build-debug" || exit;
cmake ..;
make
./grayscale_cpu ../grayscale/Rue.png
cp grayscale_cpu.png ../output
./grayscale_gpu ../grayscale/Rue.png
cp grayscale_gpu.png ../output