#ifdef __linux__
#include "Utilities/Timing/interface/LinuxCPUTime.h"
#include <iostream>
#include <fstream>
#include <limits>
#include "Utilities/General/interface/CMSexception.h"
#include "Utilities/General/interface/ioutils.h"

#include "boost/lexical_cast.hpp"

using boost::lexical_cast;

namespace Capri {
  int Proc::instanceId() {
    std::ifstream statf;
    int lpid;
    statf.open("/proc/self/stat");
    statf >> lpid;
    statf.close();
    return lpid;
  }
  
  Proc::Proc(int pid) {
    if (pid==0) pid = instanceId();
    fname = "/proc/";
    fname += lexical_cast<std::string>(pid);
    fname += "/stat";
  }
  
  void Proc::refreshTime(d& ut, d&st) {
    statf.open(fname.c_str());
    try {
      statf.seekg(0);
      int wc=0;
      while (wc<13) {
	if (!statf) {
	  std::cout << "bad "<< fname.c_str() << std::endl;
	  statf.close();
	  statf.open(fname.c_str());
	  std::cout << statf.rdbuf() << std::endl;
	  statf.close();
	  return;
	}
	statf.ignore(std::numeric_limits<int>::max(),' ');
	wc++;
      }
      statf >> ut >> st;
    }
    catch (std::exception &)  {
      statf.close();
      statf.open(fname.c_str());
    }
    
    statf.close();
    
  }
  
  ProcStat::ProcStat() { 
    refresh();
  }
  
  void ProcStat::refresh() {
    statf.open(fname.c_str());
    try {
      statf.seekg(0);
      statf >> pid >> comm >> state >> ppid >> pgrp
	    >> session >> tty  >> tpgid >> flags
	    >> minflt >> cminflt >> majflt >> cmajflt
	    >> utime >> stime >> cutime >> cstime >> counter;
    }
    catch (std::exception &)  {
      statf.close();
      statf.open(fname.c_str());
    }
    
    statf.close();
    
  }
  
}

std::ostream * LinuxElapsedTime::dout = &std::cout;

LinuxElapsedTime::~LinuxElapsedTime() {
  LinuxCPUTime end;
  out << "\n" << name << " CPU elapsed time " 
      << end.utime()-begin.utime() << "u "
      << end.stime()-begin.stime() << "s"
      << std::endl;
  
}


namespace {

  static LinuxElapsedTime totaltime("Main Thread");

}

#endif
