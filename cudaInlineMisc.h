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

#define cuiIsPowerOf2(n)      (((n-1)&n)==0)
#define cuiGetWarpsPerBlock() ((blockDim.x  >> 5) + ((blockDim.x & 31)!=0))
#define cuiGetWarpID()        (threadIdx.x >> 5)
#define cuiGetWarpTID()       (threadIdx.x & 31)
#define cuiGetGlobalTID()     (blockIdx.x * blockDim.x + threadIdx.x)
#define cuiGetGlobalWID()     (blockIdx.x * cuiGetWarpsPerBlock() + cuiGetWarpID())



inline void cuiTestMacros(){
  for(int i=0; i<1000000; i++){
    // test warps per block etc.
    {
      cvec2i blockDim;
      blockDim.x = i;
      assertPrint(cuiGetWarpsPerBlock()==(ceil_int_div(i,32)), "%d != %d\n",cuiGetWarpsPerBlock(),(ceil_int_div(i,32)));
    }
  }
}

#define ALLSYNC {__threadfence(); __syncthreads();}

inline void cuiReportMemUsage(const char* prefix=""){
  // report memory usage
  size_t avail;
  size_t total;
  cudaMemGetInfo( &avail, &total );
  size_t used = total - avail;
  printf("%s using %dMB of %dMB\n",prefix,used >> 20, total >> 20);
}

// CUDA based synchronization

#define cuiTryLock(lockPtr, success)    do { \
                                          success = (atomicExch(lockPtr, 1)==0); \
                                        } while(0)
#define cuiFreeLock(lockPtr)            do { \
                                          atomicExch(lockPtr, 0); \
                                        } while(0)


#ifdef __CUDACC__
// Get current SM ID
// from:
// http://stackoverflow.com/questions/2983553/cuda-injecting-my-own-ptx-function
__device__ __forceinline__
unsigned cuiGetSMID(void){
  unsigned ret;
  asm("mov.u32 %0, %smid;" : "=r"(ret) );
  return ret;
}
#endif

#endif
