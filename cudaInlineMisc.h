#ifndef cudaInlineMisc_h
#define cudaInlineMisc_h

#include "maths.h"

#define cudaKernelLaunch(kernelName, blockSize, nTotalElements,...)  do {   \
                                                                         dim3 blk(blockSize,1,1); \
                                                                         dim3 grd(ceil_int_div(nTotalElements,blk.x),1,1); \
                                                                         kernelName<<<grd,blk>>>(__VA_ARGS__); \
                                                                     }while(0)

#define GLOBAL_TID (blockIdx.x * blockDim.x + threadIdx.x)

#define ALLSYNC {__threadfence(); __syncthreads();}

#endif
