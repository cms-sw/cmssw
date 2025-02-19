#include "Utilities/Timing/interface/PentiumTimer.h"
#include <iostream>
#include <fstream>
#include <string>
#include <algorithm>
#include <iterator>


PentiumTime::OneTick::~OneTick(){}

PentiumTime::OneTick::OneTick() {
#ifdef __APPLE__
  // FIXME: PentiumTime not supported on mac.
  abort(); 
#else
  std::string input; 
  {
    std::ifstream cpuinfo("/proc/cpuinfo");
    cpuinfo.unsetf( std::ios::skipws );
    std::istream_iterator<char> sbegin(cpuinfo),send;
    copy(sbegin,send,inserter(input,input.end()));
    cpuinfo.close();
  }
  size_t i = input.find("cpu MHz");
  if (i==std::string::npos) {
    std::cout << "/proc/cpuinfo does not contain cpu speed..." << std::endl;
    one = 1.;
    return;
  }
  i = input.find(":",i);  
  one = 1.e-6/atof(input.substr(i+1,input.find("/n",i)-i).c_str());
#endif
}
