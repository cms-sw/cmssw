#include <cuda_runtime.h>

#include "Test/KernelFunction/interface/function.h"
#include "Test/KernelFunction/interface/kernel.h"

namespace KernelFunction
{

  __global__
  void kernel()
  {
    KernelFunction::function();
  }

}
