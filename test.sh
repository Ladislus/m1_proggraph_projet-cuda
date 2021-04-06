### Dossiers
echo 'Checking for directory "cmake-build-debug"...';
if [ ! -d 'cmake-build-debug' ]; then
  mkdir 'cmake-build-debug';
  echo 'Created !';
else
  echo 'OK';
fi
echo;
echo 'Checking for directory "output"...';
if [ ! -d 'output' ]; then
  mkdir 'output';
  echo 'Created !';
else
    echo 'OK';
fi
echo;
echo 'Moving to "cmake-build-debug"...';
cd 'cmake-build-debug' || exit;
echo 'Done';
echo;

### CMake / Compile
echo 'Generating CMake configuration';
cmake ..;
echo 'Cmake done';
echo;
echo 'Compiling';
make;
echo 'Program compiled';
echo;

### Grayscale
echo 'Executing "grayscale_cpu"';
time ./grayscale_cpu ../grayscale/Rue.png;
echo 'Copying outpout file';
cp grayscale_cpu.png ../output;
echo;
echo 'Executing "grayscale_gpu"';
time ./grayscale_gpu ../grayscale/Rue.png;
echo 'Copying outpout file';
cp grayscale_gpu.png ../output;
echo;

### ASCII
echo 'Executing "ascii_cpu"';
time ./ascii_cpu ../asciiart/pokeball.png;
echo 'Copying outpout file';
cp ascii_cpu.txt ../output;
echo;
echo 'Executing "ascii_gpu"';
time ./ascii_gpu ../asciiart/pokeball.png;
echo 'Copying outpout file';
cp ascii_gpu.txt ../output;
echo;

### CONVOLUTION
echo 'Executing "ascii_cpu"';
time ./convolution_cpu ../convolution/Celeste.png;
echo 'Copying outpout file';
cp convolution_gpu.png ../output;
echo;
echo 'Executing "ascii_gpu"';
time ./convolution_gpu ../convolution/Celeste.png;
echo 'Copying outpout file';
cp convolution_gpu.png ../output;
echo;

### Fin
echo 'TESTS DONE';