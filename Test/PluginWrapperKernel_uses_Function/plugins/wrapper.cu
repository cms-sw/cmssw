#include <cuda_runtime.h>

#include "kernel.h"
#include "wrapper.h"

namespace PluginWrapperKernel_uses_Function {

  __host__
  void wrapper()
  {
    PluginWrapperKernel_uses_Function::kernel<<<1, 1>>>();
    cudaDeviceSynchronize();
  }

}
