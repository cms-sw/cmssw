#ifndef UTILITIES_TIMING_GENTIMER_H
#define UTILITIES_TIMING_GENTIMER_H
//
//   V 0.0 
//
#include <iosfwd>

/** a generic Time interval
 */
template<class Time>
class GenTimeInterval {
public:
  typedef typename Time::IntervalType IntervalType;
  typedef typename Time::IntervalType T;

public:
  //
  GenTimeInterval(IntervalType i=0) : it(i) {}

  //
  GenTimeInterval & operator =(IntervalType i) { it=i; return *this;}

  //
  operator IntervalType & () { return it;}

  //
  operator const IntervalType & () const { return it;}

  //
  const IntervalType & ticks() const  { return it;}

  //
  double seconds() const { return Time::oneTick()*it; }
  
  //
  double microseconds() const { return 1.e6*seconds();}

private:
  IntervalType it;
};


template<class Time>
std::ostream & operator<<(std::ostream & o, const GenTimeInterval<Time> & t) {
  return o << t.seconds() << " seconds";
}

/** a templated generic timer
 */
template<class Time>
class GenTimer {
public:
  typedef typename Time::TimeInterval TimeInterval;
  typedef typename Time::IntervalType IntervalType;

  typedef GenTimer<Time> self;

public:

  struct Bias {
    typedef GenTimer<Time> IT;
    double mes;
    IntervalType met;
    Bias(unsigned int n=5000) {
      mes=0.;
      met=0;
      if(n==0) return;
      IT it; 
      for (unsigned int i=0; i<n;i++) {
	it.start();it.stop();
      }
      mes = it.lap().seconds()/double(n);
      met = it.lap().ticks()/IntervalType(n);
    }
  };

  static double bias(bool insec=true, unsigned int n=5000) {
    static Bias it(n);
    return insec ? it.mes : double(it.met) ;
  }


  static double ticksInSec() { return Time::oneTick();}

public:
  /// constructor
  GenTimer() : elapsed(0),  running_(0), pid(0) {}
  /// from known context
  GenTimer(int ipid) : elapsed(0),  running_(0), pid(ipid) {}

  /// destructor
  ~GenTimer(){}

  //
  inline bool running() const { return running_>0;}

  //
  inline void reset() { 
    if (running()) elapsed=-Time::time(pid);
    else elapsed=0;
  }

  //
  inline void start() {
    running_++;
    if (running_==1) elapsed-=Time::time(pid);
  }

  //
  inline void stop()  {
    if (running_==1) elapsed+=Time::time(pid);
    if (running_>0) running_--;
  }

  inline void forceStop() {
    if (running_==0) return;
    running_ =1;
    stop();
  }

  //
  inline typename Time::TimeInterval lap() const { 
    if (running()) return (elapsed+Time::time(pid));
    return elapsed;
  }

private:

  typename Time::IntervalType elapsed;
  int running_;
  int pid;

};

struct DummyTime {

  struct OneTick {
    OneTick() : one(1.0) {}
    
    double one;
  };
  
  static double oneTick() {
    static OneTick local;
    return local.one;
  };


  typedef GenTimeInterval<DummyTime> TimeInterval;
  typedef long long int IntervalType;
  typedef long long int TimeType;

  inline static TimeType time() { return 0;}
  inline static TimeType time(int) {return 0;}
};

typedef GenTimer<DummyTime> DummyTimer; 

#endif // UTILITIES_TIMING_GENTIMER_H
