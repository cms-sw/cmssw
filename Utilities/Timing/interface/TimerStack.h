#ifndef Utilities_Timing_TimerStack_h
#define Utilities_Timing_TimerStack_h 1
// Package:    UtiBlities/Timing
// Class:      TimerStack
// 
/*

 Description: Tool to manipulate multiple timers

 */
//
// Original Author:  Dmytro Kovalskyi
// $Id: TimerStack.h,v 1.1 2007/04/10 06:42:57 dmytro Exp $

#include "Utilities/Timing/interface/TimingReport.h"
#include <stack>


class TimerStack
{
 public:
   /// Types of time interval measurements used in TimeReport:
   /// - cpu clock counter called "real". It is fast, high 
   ///   precision, but CPU load from other processes can affect 
   ///   the result
   /// - system process time (/proc/"process id"/stat) called 
   ///   "cpu", is slow (few ms per call), but CPU load from 
   ///   other processes is factored out.
   /// FastMonitoring uses only the first one and DetailedMonitoring
   /// uses both. Recommend usage:
   /// - slow code fragments (> 10 ms) - DetailedMonitoring
   /// - fast code fragments (microseconds) - FastMonitoring
   ///
   enum Type { DetailedMonitoring, FastMonitoring };
   enum Status { AlwaysActive, Disableable };
   
   /// TTimer is a container for a timer name and associated timers (TimeReport::Item's)
   class Timer {
    public:
      Timer(const std::string& name):
        first_( &( (*TimingReport::current())["firstcall_"+name] ) ),
        main_( &( (*TimingReport::current())[name] ) ),
        name_(name){}
      TimingReport::Item& first() { return *first_; }
      TimingReport::Item& main() { return *main_; }
    private:
      Timer(){}
      TimingReport::Item* first_;
      TimingReport::Item* main_;
      std::string name_;
   };

   TimerStack():status_(AlwaysActive){}
   TimerStack(Status status):status_(status) {}

   ~TimerStack() { clear_stack(); }
   
   /// measure time to perform a number of floating point multiplications (FLOPs)
   void benchmark( std::string name, int n = 1000000);
   
   /// start a timer and add it to the stack
   void push( std::string name, Type type = FastMonitoring );
   void push( Timer&, Type type = FastMonitoring );
   /// stop the last timer and remove it from the stack
   void pop();
   /// stop all timers in the stack and clear it.
   void clear_stack();
   /// stop the last timer and remove it from the stack,
   /// than start a new timer and add it to the stack
   /// (convinient way of timing sequential code fragments)
   void pop_and_push( std::string name, Type type = FastMonitoring );
   void pop_and_push( Timer&, Type type = FastMonitoring );
   
   /// get access to the cpu clock counter on Intel and AMD cpus
   PentiumTimeType pentiumTime() {return rdtscPentium();}
	
 private:
   std::stack<TimeMe*> stack;
   Status status_;
};

#define FastTimerStackPush(timer, name) \
   { static TimerStack::Timer t = TimerStack::Timer(std::string(name)); \
     timer.push( t );}
#endif
