#ifndef TrackAssociator_TimerStack_h
#define TrackAssociator_TimerStack_h 1
#include "Utilities/Timing/interface/TimingReport.h"
#include <stack>
class TimerStack
{
 public:
   enum Type { DetailedMonitoring, FastMonitoring };
   enum Status { AlwaysActive, Disableable };
   
   TimerStack():status_(AlwaysActive){}
   TimerStack(Status status):status_(status) {}

   ~TimerStack() { clean_stack(); }
   
   // has no effect if status_ is AlwaysActive
   void disableAllTimers(){ disabled_ = true; }
   void enableAllTimers(){ disabled_ = false; }
   
   // measure time to perform a number of floating point multiplications (FLOPs)
   void benchmark( std::string name, int n = 1000000);

   void push( std::string name, Type type = DetailedMonitoring );
   void pop();
   void clean_stack();
   void pop_and_push(std::string name, Type type = DetailedMonitoring );
   
   PentiumTimeType pentiumTime() {return rdtscPentium();}
	
 private:
   std::stack<TimeMe*> stack;
   Status status_;
   static bool disabled_;
};
#endif
