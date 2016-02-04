#ifndef UTILITIES_TIMING_LINUXCPUTIME_H
#define UTILITIES_TIMING_LINUXCPUTIME_H
//
//   V 0.0 
//


#include <iosfwd>
#include <string>
#include <fstream>
#include "ctime"
#include "Utilities/Timing/interface/GenTimer.h"

/**
 */
class LinuxCPUTime {
public:

   typedef double TimeType;


  /// constructor
  explicit LinuxCPUTime(int pid=0): 
    utime_(std::clock()/CLOCKS_PER_SEC), stime_(0) {
  }

  /// destructor
  ~LinuxCPUTime(){}

  ///
  inline TimeType utime() const { return utime_;}
  inline TimeType stime() const { return stime_;}
  inline TimeType cputime() const { return utime_+stime_;}
  inline TimeType operator()() const { return cputime();}

private:
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
  
  static double oneTick() {
    static OneTick local;
    return local.one;
  };


  typedef GenTimeInterval<LCPUTime> TimeInterval;
  typedef LinuxCPUTime::TimeType IntervalType;
  typedef LinuxCPUTime::TimeType TimeType;

  inline static TimeType time() { LinuxCPUTime a; return a();}
  inline static TimeType time(int pid) { LinuxCPUTime a(pid); return a();}

};

typedef GenTimer<LCPUTime> LinuxCPUTimer; 


#endif // UTILITIES_TIMING_LINUXCPUTIME_H
