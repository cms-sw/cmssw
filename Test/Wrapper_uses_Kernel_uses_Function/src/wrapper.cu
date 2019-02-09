#include <cuda_runtime.h>

#include "Test/Kernel_uses_Function/interface/kernel.h"
#include "Test/Wrapper_uses_Kernel_uses_Function/interface/wrapper.h"

namespace Wrapper_uses_Kernel_uses_Function {

  __host__
  void wrapper()
  {
    Kernel_uses_Function::kernel<<<1, 1>>>();
    cudaDeviceSynchronize();
  }

}
