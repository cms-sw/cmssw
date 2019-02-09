#include <cuda_runtime.h>

#include "function.h"
#include "kernel.h"

namespace PluginWrapperKernelFunction
{

  __global__
  void kernel()
  {
    PluginWrapperKernelFunction::function();
  }

}
