#include <cuda_runtime.h>

#include "kernel.h"
#include "wrapper.h"

namespace BinaryWrapperKernelFunction {

  __host__
  void wrapper()
  {
    BinaryWrapperKernelFunction::kernel<<<1, 1>>>();
    cudaDeviceSynchronize();
  }

}
