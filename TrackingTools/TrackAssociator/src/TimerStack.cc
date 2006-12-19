#include "TrackingTools/TrackAssociator/interface/TimerStack.h"
bool TimerStack::disabled_ = false;
void TimerStack::push( std::string name, Type type ){
   bool linuxCpuOn = (type == DetailedMonitoring);
   if ( status_==Disableable && disabled_ ) return;
   if( (*TimingReport::current())["firstcall_"+name].counter == 0)
     stack.push(new TimeMe("firstcall_"+name,linuxCpuOn));
   else
     stack.push(new TimeMe(name,linuxCpuOn));
}

void TimerStack::pop( ){
   if (!stack.empty()) {
      delete stack.top();
      stack.pop();
   }
}

void TimerStack::clean_stack(){
   while(!stack.empty()) pop();
}
   
void TimerStack::pop_and_push(std::string name, Type type ) {
   if ( status_==Disableable && disabled_ ) return;
   pop();
   push(name,type);
}

void TimerStack::benchmark( std::string name, int n )
{
   push(name,FastMonitoring);
   float a(1.0009394);
   float b(0.000123);
   for(int i=0; i < n; i++) b *= a;
   pop();
}
