echo 'Checking for directory "cmake-build-debug"...';
if [ ! -d 'cmake-build-debug' ]; then
  mkdir 'cmake-build-debug';
  echo 'Created !';
else
  echo 'OK';
fi
echo 'Checking for directory "output"...';
if [ ! -d 'output' ]; then
  mkdir 'cmake-build-debug';
  echo 'Created !';
else
    echo 'OK';
fi
echo 'Moving to "cmake-build-debug"...';
cd 'cmake-build-debug' || exit;
echo 'Done';
echo 'Generating CMake configuration';
cmake ..;
echo 'Cmake done';
echo 'Compiling';
make;
echo 'Cmake done';
echo 'Executing "grayscale_cpu"';
time ./grayscale_cpu ../grayscale/Rue.png;
echo 'Copying outpout file';
cp grayscale_cpu.png ../output;
echo 'Executing "grayscale_gpu"';
time ./grayscale_gpu ../grayscale/Rue.png;
echo 'Copying outpout file';
cp grayscale_gpu.png ../output;
echo 'TESTS DONE';