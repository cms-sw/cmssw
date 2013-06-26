#include "Utilities/Timing/interface/TimingReport.h"
#include <algorithm>
#include <iostream>
#include <ostream>
#include <iomanip>
#include <map>

double TimingReport::Item::realsec() const {
  return std::max(0.,stopwatch.lap().seconds()-counter*PentiumTimer::bias());
}

double TimingReport::Item::realticks() const {
  return std::max(0.,stopwatch.lap().ticks()-counter*PentiumTimer::bias(false));
}

double TimingReport::Item::cpusec() const {
  return std::max(0.,cpuwatch.lap().seconds()-counter*(LinuxCPUTimer::bias()+PentiumTimer::bias()));
}


TimingReport * TimingReport::current() {
  static TimingReport * currentTimingReport=0;

  if (currentTimingReport==0) currentTimingReport = new TimingReport();
  return currentTimingReport;

}

TimingReport::TimingReport() : on(true), inTicks_(false) {
    //std::cout << "Creating a new Timing Report" << std::endl;    
    //std::cout <<"StopWatch bias " << PentiumTimer::bias()<<std::endl;
    //std::cout <<"CPUWatch bias " << LinuxCPUTimer::bias()<<std::endl;
}

void TimingReport::switchOn(bool ion) {
    if (on==ion) return;
    on = ion;
    //std::cout << "switching Timing Report " 
    //        << (on ? "on" : "off") << std::endl;
  SMAP::iterator p = registry.begin();
  SMAP::iterator e = registry.end();
  for (;p!=e; ++p) (*p).second.switchOn(on);
}

TimingReport::~TimingReport() {  
    if (!on) return;
    dump(std::cout);
}

void TimingReport::dump(std::ostream & ico, bool active) {  
    /// make the output sorted
  typedef std::map<std::string, Item *, std::less<std::string> > LMAP;
  LMAP lreg;
  {
    SMAP::iterator p = registry.begin();
    SMAP::iterator e = registry.end();
    while (p!=e) { 
      if ( (*p).second.on && (!active||(*p).second.active()) )
	lreg[(*p).first] = &(*p).second; 
      ++p;
    }
  }

  std::ostream co(ico.rdbuf());
  size_t namew = 20;
  co << "\n";
  if (active) co << "Active ";
  co << "Timing Report  (in "
     << (inTicks() ? "ticks" : "seconds") << ")\n" << std::endl;
  LMAP::iterator p = lreg.begin();
  LMAP::iterator e = lreg.end();
  while (p!=e) { namew = std::max(namew,(*p).first.size()); ++p;}
  p = lreg.begin();
  while (p!=e) { 
    co.setf(std::ios::left,std::ios::adjustfield);
    if (inTicks())
      co << std::setiosflags(std::ios::showpoint)
	 << std::setprecision(3);
    else
      co << std::setiosflags(std::ios::showpoint | std::ios::fixed)
	 << std::setprecision(3);

    co << std::setw(namew) << (*p).first.c_str() << " "; 
    co.setf(std::ios::right,std::ios::adjustfield);
    co << std::setw(10) << (*(*p).second).counter << "   "; 
    co << std::setw(10) 
       <<  (inTicks() ? (*(*p).second).realticks() 
	    : (*(*p).second).realsec() ) << " (real)"; 
#ifdef __linux__
    co << "   " << std::setw(10)
       << (inTicks() ? 1./PentiumTimer::ticksInSec() : 1.)*(*(*p).second).cpusec() << " (cpu)"; 
#endif
    co << std::endl;
    ++p;
  }
  co << "\n" << std::endl;
}
