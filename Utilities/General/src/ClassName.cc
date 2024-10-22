#include "Utilities/General/interface/ClassName.h"
#include <cxxabi.h>
#include <cstring>

Demangle::Demangle(const char* sc) : demangle(nullptr) {
  if (sc == nullptr)
    return;
  int status;
  demangle = abi::__cxa_demangle(sc, nullptr, nullptr, &status);
  if (status == 0)
    return;
  demangle = nullptr;
  if (status == -1)
    throw std::bad_alloc();
  else if (status == -2) {
    demangle = strdup(sc);
  }
}
