#include <cuda_runtime.h>

#include "Test/Function/interface/function.h"
#include "kernel.h"

namespace PluginWrapperKernel_uses_Function
{

  __global__
  void kernel()
  {
    Function::function();
  }

}
