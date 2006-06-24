#ifndef TrackAssociator_TimerStack_h
#define TrackAssociator_TimerStack_h 1
#include "Utilities/Timing/interface/TimingReport.h"
#include <stack>
class TimerStack
{
 public:
   ~TimerStack()
     {
	clean_stack();
     }
   
   void push(std::string name){
      if( (*TimingReport::current())["firstcall_"+name].counter == 0)
	stack.push(new TimeMe("firstcall_"+name));
      else
	stack.push(new TimeMe(name));
   }
   
   void pop(){
      if (!stack.empty()) {
	 delete stack.top();
	 stack.pop();
      }
   }

   void clean_stack(){
      while(!stack.empty()) pop();
   }
   
   void pop_and_push(std::string name) {
      pop();
      push(name);
   }
   
 private:
   std::stack<TimeMe*> stack;
};
#endif
