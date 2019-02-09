#include <iostream>
#include <string_view>

#include <cuda_runtime.h>

#include "wrapper.h"

int main(int argc, char* argv[]) {
  BinaryWrapperKernel_uses_Function::wrapper();

  // check if the underlying CUDA kernel was successful
  auto result = cudaGetLastError();
  if (result == cudaSuccess)
    return 0;

  // get the command name
  std::string_view fullname(argv[0]);
  // strip any leading directories
  auto pos = fullname.find_last_of('/');
  pos = (pos != std::string_view::npos) ? pos + 1 : 0;

  // retrieve the CUDA error and description
  const char* error   = cudaGetErrorName(result);
  const char* message = cudaGetErrorString(result);
  std::cerr << fullname.substr(pos) << ": " << error << ": " << message << std::endl;

  return result;
}
