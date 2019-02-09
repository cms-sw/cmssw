#include <cstdio>
#include <cuda_runtime.h>

#include "Test/WrapperKernelFunction/interface/function.h"

namespace WrapperKernelFunction
{

  __device__
  void function()
  {
    printf("block %d,%d,%d, thread %d,%d,%d\n", blockIdx.x, blockIdx.y, blockIdx.z, threadIdx.x, threadIdx.y, threadIdx.z);
  }

}
