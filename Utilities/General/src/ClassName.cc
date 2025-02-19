#include "Utilities/General/interface/ClassName.h"
#include <cxxabi.h>
#include <cstring>

Demangle::Demangle(const char * sc) : demangle(0) {
  if (sc==0) return;
  int status;
  demangle = abi::__cxa_demangle(sc,  0, 0, &status);
  if(status == 0) return;  
  demangle = 0;
  if(status == -1)
    throw std::bad_alloc();
  else if(status == -2) {
    demangle = strdup(sc);
  }
}

