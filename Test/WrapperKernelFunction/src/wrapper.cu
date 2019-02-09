#include <cuda_runtime.h>

#include "Test/WrapperKernelFunction/interface/kernel.h"
#include "Test/WrapperKernelFunction/interface/wrapper.h"

namespace WrapperKernelFunction {

  __host__
  void wrapper()
  {
    WrapperKernelFunction::kernel<<<1, 1>>>();
    cudaDeviceSynchronize();
  }

}
