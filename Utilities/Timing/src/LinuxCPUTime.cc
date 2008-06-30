#include "Utilities/Timing/interface/LinuxCPUTime.h"
#include <iostream>
#include <fstream>
#include <limits>
#include "Utilities/General/interface/CMSexception.h"
#include "Utilities/General/interface/ioutils.h"

 
std::ostream * LinuxElapsedTime::dout = &std::cout;

LinuxElapsedTime::~LinuxElapsedTime() {
  LinuxCPUTime end;
  out << "\n" << name << " CPU elapsed time " 
      << end.utime()-begin.utime() 
      << std::endl;  
}


