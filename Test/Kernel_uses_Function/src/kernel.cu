#include <cuda_runtime.h>

#include "Test/Function/interface/function.h"
#include "Test/Kernel_uses_Function/interface/kernel.h"

namespace Kernel_uses_Function
{

  __global__
  void kernel()
  {
    Function::function();
  }

}
