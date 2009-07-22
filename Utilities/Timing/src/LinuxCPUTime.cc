#include "Utilities/Timing/interface/LinuxCPUTime.h"
#include <iostream>
 
std::ostream * LinuxElapsedTime::dout = &std::cout;

LinuxElapsedTime::~LinuxElapsedTime() {
  LinuxCPUTime end;
  out << "\n" << name << " CPU elapsed time " 
      << end.utime()-begin.utime() 
      << std::endl;  
}


