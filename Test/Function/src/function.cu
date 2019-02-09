#include <cstdio>
#include <cuda_runtime.h>

#include "Test/Function/interface/function.h"

namespace Function
{

  __device__
  void function()
  {
    printf("block %d,%d,%d, thread %d,%d,%d\n", blockIdx.x, blockIdx.y, blockIdx.z, threadIdx.x, threadIdx.y, threadIdx.z);
  }

}
