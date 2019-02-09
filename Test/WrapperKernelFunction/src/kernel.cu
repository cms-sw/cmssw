#include <cuda_runtime.h>

#include "Test/WrapperKernelFunction/interface/function.h"
#include "Test/WrapperKernelFunction/interface/kernel.h"

namespace WrapperKernelFunction
{

  __global__
  void kernel()
  {
    WrapperKernelFunction::function();
  }

}
