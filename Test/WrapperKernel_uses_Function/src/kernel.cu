#include <cuda_runtime.h>

#include "Test/Function/interface/function.h"
#include "Test/WrapperKernel_uses_Function/interface/kernel.h"

namespace WrapperKernel_uses_Function
{

  __global__
  void kernel()
  {
    Function::function();
  }

}
