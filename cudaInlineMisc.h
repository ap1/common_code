#ifndef cudaInlineMisc_h
#define cudaInlineMisc_h


//#define CUDALAUNCH(kernelName, blockSize, nTotalElements,...)

#define GLOBAL_TID (blockIdx.x * blockDim.x + threadIdx.x)

#define ALLSYNC {__threadfence(); __syncthreads();}

#endif
