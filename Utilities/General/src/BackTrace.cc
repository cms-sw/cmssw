#include "Utilities/General/interface/BackTrace.h"
#include "Utilities/General/interface/ClassName.h"
#include <iostream>
#include <cstring>

BackTrace::BackTrace() {}

void BackTrace::trace() const {
  trace(std::cout);
}


#ifdef __linux__

#include <execinfo.h>
#include <cstdlib>
#include <cstdio> 
#ifndef  __USE_GNU
#define __USE_GNU
#endif

#include <dlfcn.h>
#include <cxxabi.h>

void BackTrace::trace(std::ostream & out) const{
  static const unsigned int bsize(1024U);
  char buffer [bsize];
  void * ltrace [MAX_BACKTRACE_DEPTH];
  int  depth = backtrace (ltrace, MAX_BACKTRACE_DEPTH);
  if (depth>MAX_BACKTRACE_DEPTH) {
    out << "Error in backtrace" << std::endl;
    return;
  }
  for (int n = 0; n < depth; ++n) {
    unsigned long   addr = (unsigned long) ltrace[n];
    Dl_info         info;
    
    if (dladdr (ltrace[n], &info) && info.dli_fname && info.dli_fname[0]) {
      const char          *libname = info.dli_fname;
      unsigned long       symaddr = (unsigned long) info.dli_saddr;
      if (info.dli_sname && info.dli_sname[0]) {
	Demangle ln(info.dli_sname);
	bool                gte = (addr >= symaddr);
	unsigned long       diff = (gte ? addr - symaddr : symaddr - addr);
	sprintf (buffer, " 0x%08lx %.100s %s 0x%lx [%.100s]\n",
		 addr, ln(), gte ? "+" : "-", diff, libname);
      }	else 
	sprintf (buffer, " 0x%08lx <unknown function> [%.100s]\n", addr, libname);
    } else {
      sprintf (buffer, " 0x%08lx <unknown function>\n", addr);
    }
    if (::strlen (buffer) > bsize) {
      out << "Error in backtrace" << std::endl;
      return;
    }
    out.write (buffer, ::strlen (buffer));
    out.flush();
  }
#ifndef CMS_CHAR_STREAM
  out << std::ends;
#endif
}  
#else
void BackTrace::trace(std::ostream & out) const {}
#endif

