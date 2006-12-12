#ifndef UTILITIES_TIMING_LINUXCPUTIME_H
#define UTILITIES_TIMING_LINUXCPUTIME_H
//
//   V 0.0 
//
#ifdef __linux__

#include <iosfwd>
#include <string>
#include <fstream>
#include "Utilities/General/interface/Proc.h"
#include "Utilities/Timing/interface/GenTimer.h"

/**
 */
class LinuxCPUTime {
public:

  static Capri::ProcStat & procStat(){
    static Capri::ProcStat local;
    return local;
  }

  //  typedef long long int TimeType;
  typedef Capri::Proc::d TimeType;

  /// constructor
  explicit LinuxCPUTime(int pid=0): proc(pid) {
    proc.refreshTime(utime_,stime_);
  }

  LinuxCPUTime(const Capri::ProcStat & p) : utime_(p.utime), stime_(p.stime){}
  /// destructor
  ~LinuxCPUTime(){}

  ///
  inline TimeType utime() const { return utime_;}
  inline TimeType stime() const { return stime_;}
  inline TimeType cputime() const { return utime_+stime_;}
  inline TimeType operator()() const { return cputime();}

private:
  Capri::Proc proc;
  TimeType utime_;
  TimeType stime_;
};

class LinuxElapsedTime {
public:
  static std::ostream * dout;
  LinuxElapsedTime(const std::string & iname="Total",
		   std::ostream & iout=*dout) : out(iout), name(iname){}
  ~LinuxElapsedTime();

private:
  std::ostream & out;
  std::string name;
  LinuxCPUTime begin;
};


struct LCPUTime {

  struct OneTick {
    OneTick() : one(0.01) {}
    
    double one;
  };
  
  static OneTick & oneTick() {
    static OneTick local;
    return local;
  };


  typedef GenTimeInterval<LCPUTime> TimeInterval;
  typedef LinuxCPUTime::TimeType IntervalType;
  typedef LinuxCPUTime::TimeType TimeType;

  inline static TimeType time() { LinuxCPUTime a; return a();}
  inline static TimeType time(int pid) { LinuxCPUTime a(pid); return a();}

};

typedef GenTimer<LCPUTime> LinuxCPUTimer; 

#endif // __linux__

#endif // UTILITIES_TIMING_LINUXCPUTIME_H
