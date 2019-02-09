#include <cuda_runtime.h>

#include "Test/KernelFunction/interface/kernel.h"
#include "wrapper.h"

namespace PluginWrapper_uses_KernelFunction {

  __host__
  void wrapper()
  {
    KernelFunction::kernel<<<1, 1>>>();
    cudaDeviceSynchronize();
  }

}
