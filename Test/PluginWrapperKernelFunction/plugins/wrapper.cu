#include <cuda_runtime.h>

#include "kernel.h"
#include "wrapper.h"

namespace PluginWrapperKernelFunction {

  __host__
  void wrapper()
  {
    PluginWrapperKernelFunction::kernel<<<1, 1>>>();
    cudaDeviceSynchronize();
  }

}
