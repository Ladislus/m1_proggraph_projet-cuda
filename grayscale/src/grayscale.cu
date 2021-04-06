__global__ void grayscale( unsigned char * rgb, unsigned char * g, std::size_t w, std::size_t h) {
    auto tidx = blockIdx.x * blockDim.x + threadIdx.x;
    auto tidy = blockIdx.y * blockDim.y + threadIdx.y;

    if ( w < tidx && h < tidy) {
        g[ j * cols + i ] = ( 307 * rgb[ 3 * ( j * cols + i ) ] } +
                                                                          604 * rgb[ 3 * ( j * cols + i ) + 1 ]} +
113 * rgb[ 3 * ( j  * cols + i ) + 2 ]
) / 1024;
}
}