#include <cuda_runtime.h>

#include "Test/Kernel_uses_Function/interface/kernel.h"
#include "wrapper.h"

namespace PluginWrapper_uses_Kernel_uses_Function {

  __host__
  void wrapper()
  {
    Kernel_uses_Function::kernel<<<1, 1>>>();
    cudaDeviceSynchronize();
  }

}
