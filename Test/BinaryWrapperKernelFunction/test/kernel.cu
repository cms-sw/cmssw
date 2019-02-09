#include <cuda_runtime.h>

#include "function.h"
#include "kernel.h"

namespace BinaryWrapperKernelFunction
{

  __global__
  void kernel()
  {
    BinaryWrapperKernelFunction::function();
  }

}
