#include <cstdio>
#include <cuda_runtime.h>

#include "Test/KernelFunction/interface/function.h"

namespace KernelFunction
{

  __device__
  void function()
  {
    printf("block %d,%d,%d, thread %d,%d,%d\n", blockIdx.x, blockIdx.y, blockIdx.z, threadIdx.x, threadIdx.y, threadIdx.z);
  }

}
