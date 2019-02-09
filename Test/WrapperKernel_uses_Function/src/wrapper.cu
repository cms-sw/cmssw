#include <cuda_runtime.h>

#include "Test/WrapperKernel_uses_Function/interface/kernel.h"
#include "Test/WrapperKernel_uses_Function/interface/wrapper.h"

namespace WrapperKernel_uses_Function {

  __host__
  void wrapper()
  {
    WrapperKernel_uses_Function::kernel<<<1, 1>>>();
    cudaDeviceSynchronize();
  }

}
