#include <cuda_runtime.h>

#include "Test/KernelFunction/interface/kernel.h"
#include "Test/Wrapper_uses_KernelFunction/interface/wrapper.h"

namespace Wrapper_uses_KernelFunction {

  __host__
  void wrapper()
  {
    KernelFunction::kernel<<<1, 1>>>();
    cudaDeviceSynchronize();
  }

}
