#ifndef UTILITIES_TIMING_PENTIUMTIMER_H
#define UTILITIES_TIMING_PENTIUMTIMER_H
//
//
//   V 0.0 
//

// Linux only:
#ifdef __linux__

#include "Utilities/Timing/interface/GenTimer.h"

typedef unsigned long long int PentiumTimeType;
typedef long long int PentiumTimeIntervalType;

extern "C" inline PentiumTimeType rdtscPentium() {
  PentiumTimeType x;
  __asm__ volatile (".byte 0x0f, 0x31" : "=A" (x));
  return x;
}



struct PentiumTime {

  struct OneTick {
    OneTick();
    ~OneTick();
    
    double one;
  };
  
  static double oneTick() {
    static OneTick local;
    return local.one;
  };

  typedef GenTimeInterval<PentiumTime> TimeInterval;
  typedef PentiumTimeIntervalType IntervalType;
  typedef PentiumTimeType TimeType;

  inline static TimeType time() { return rdtscPentium();}
  inline static TimeType time(int) { return rdtscPentium();}

};


/**  a timer valid only on Linux Pentium
 */

typedef GenTimer<PentiumTime> PentiumTimer; 


template<>
inline 
GenTimer<PentiumTime>::GenTimer() : elapsed(0),  running_(0), pid(0) {}

#endif //  __linux__
#endif // UTILITIES_TIMING_PENTIUMTIMER_H
