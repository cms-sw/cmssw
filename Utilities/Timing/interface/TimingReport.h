#ifndef UTILITIES_TIMING_TIMINGREPORT_H
#define UTILITIES_TIMING_TIMINGREPORT_H
//
//  V 1.1  01/09/2000
//    Fast timing on Pentium
//    on/off control introduced
//    fixed format output
//  V 1.2  21/02/2001
//    cpu time added
//    not thread safe yet...

#include <string>
#include <map>
#include <iosfwd>

#include "Utilities/Timing/interface/BaseEvent.h"
#include "Utilities/Timing/interface/PentiumTimer.h"
#include "Utilities/Timing/interface/LinuxCPUTime.h"

/*  a class to manage Timing
**/
class TimingReport {
public:
  typedef BaseEvent< std::pair<double,double> > ItemObserver;

  class Item {
    typedef BaseEvent< std::pair<double,double> > MyObserver;
  public:
    Item() : on(true), cpuon(true), counter(0), o(0){}
    Item & switchOn(bool ion) {on=ion; return *this;} 
    Item & switchCPU(bool ion) {cpuon=ion; return *this;} 
    void start() { if (on) {counter++; if (cpuon) cpuwatch.start(); stopwatch.start(); }}
    void stop(){ 
      if (on) {
	stopwatch.stop(); 
	if (cpuon) cpuwatch.stop();
	if (active()) return; 
	if (o) (*o)(std::pair<double,double>(stopwatch.lap().seconds(),
					cpuwatch.lap().seconds()));
      }
    }
  public:
    bool active() const { return stopwatch.running();}
    void setObs(MyObserver * io) { o=io;}
    double realsec() const;
    double realticks() const;
    double cpusec() const;
  public:
    bool on;
    bool cpuon;
    int counter;
    PentiumTimer stopwatch;
    LinuxCPUTimer cpuwatch;
    MyObserver * o;

  };

public:
  static TimingReport * current();

protected:

  typedef std::map< std::string, Item, std::less<std::string> > SMAP;

  TimingReport();

public:
  ~TimingReport();

  ///
  void dump(std::ostream & co, bool active=false);

  /// report in ticks
  bool & inTicks() { return inTicks_;}

  /// switch all on
  void switchOn(bool ion);

  /// switch one ion
  void switchOn(const std::string& name, bool ion) {
    registry[name].switchOn(ion);
  }

  void start(const std::string& name) {
    if(on) registry[name].start();
  }
  void stop(const std::string& name) {
    if (on) registry[name].stop();
  }
  
  Item & operator[](const std::string& name) {
    SMAP::iterator p = registry.find(name);
    if (p!=registry.end()) return (*p).second;
    return make(name);
  }
  
  const Item & operator[](const std::string& name) const {
    SMAP::const_iterator p = registry.find(name);
    if (p!=registry.end()) return (*p).second;
    return const_cast<TimingReport*>(this)->make(name);
  }

  Item & make(const std::string& name) {
    return registry[name].switchOn(on);
  }

  const bool & isOn() const { return on;} 

private:
  
  bool on;
  bool inTicks_;
  SMAP registry;


};

/** a class to time a "scope" to be used as a "Sentry".
Just create a TimeMe object giving it a name;
exiting the scope the object will be deleted;
the constuctor starts the timing.
the destructor stops it.
 */
class TimeMe{

public:
   ///
  explicit TimeMe(const std::string& name, bool cpu=true) :
    item((*TimingReport::current())[name]) {
    item.switchCPU(cpu);
    item.start();
  }

  explicit TimeMe(TimingReport::Item & iitem, bool cpu=true) :
    item(iitem) {
    item.switchCPU(cpu);
    item.start();
  }

  std::pair<double,double> lap() const { 
    return std::pair<double,double>(item.stopwatch.lap().seconds(),
			       item.cpuwatch.lap().seconds());
  }
 
  ///
  ~TimeMe() {
    item.stop();
  }
  
private:
  
  TimingReport::Item & item;
  
};

#endif // UTILITIES_TIMING_TIMINGREPORT_H
