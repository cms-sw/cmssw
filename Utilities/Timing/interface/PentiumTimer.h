#ifndef UTILITIES_TIMING_PENTIUMTIMER_H
#define UTILITIES_TIMING_PENTIUMTIMER_H
//
//
//   V 0.0 
//

#include "Utilities/Timing/interface/GenTimer.h"

typedef unsigned long long int PentiumTimeType;
typedef long long int PentiumTimeIntervalType;

#if defined(__x86_64__) || defined(__i386__)
extern "C" inline PentiumTimeType rdtscPentium() {
  PentiumTimeType x;
  // Works only for x86 machines in protected mode.
  __asm__ volatile (".byte 0x0f, 0x31" : "=A" (x));
  return x;
}
#elif defined(__arm__)
#warning PentiumTimeType rdtscPentium() is not yet implemented for ARM architecture. Default return 0.
extern "C" inline PentiumTimeType rdtscPentium() {
  return 0;
}
#else
#error PentiumTimeType rdtscPentium() is not implemented for your CPU architecture.
#endif



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

#endif // UTILITIES_TIMING_PENTIUMTIMER_H
