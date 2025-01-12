#ifndef __PARAMETERS_H_

#define __PARAMETERS_H_

#include <stdio.h>
#include <stdlib.h>

#define MAX_THREADS_PER_BLOCK 1024
#define MAX_BLOCK_SIDE 32

// Max allowed dimension for a tile (larger tiles make more advantage of shared memory)
#define tileDim dim3(32, 32)

// If enabled, main will still have float matrices, but the values will be actually integers
// (stored in a float variable). This will also affect print(...), to display matrices in
// a more compact way
// #define TEST_intValues

#endif
