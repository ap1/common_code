#ifndef cudaInlineMisc_h
#define cudaInlineMisc_h

#include "maths.h"

#define cuiKernelLaunch(kernelName, blockSize, nTotalElements,...)  do {   \
                                                                         dim3 blk(blockSize,1,1); \
                                                                         dim3 grd(ceil_int_div(nTotalElements,blk.x),1,1); \
                                                                         kernelName<<<grd,blk>>>(__VA_ARGS__); \
                                                                     }while(0)


#define cuiMallocArray(    memptr, arrcount, datatype) do {cudaMalloc((void**)memptr, arrcount*sizeof(datatype));}while(0)
#define cuiMallocSingleton(memptr,           datatype) do {cudaMalloc((void**)memptr,           1*sizeof(datatype));}while(0)

//cudaMalloc((void**)&rstbins.VSPrimCount,              1 * sizeof(int));

#define GLOBAL_TID (blockIdx.x * blockDim.x + threadIdx.x)

#define ALLSYNC {__threadfence(); __syncthreads();}

#endif
