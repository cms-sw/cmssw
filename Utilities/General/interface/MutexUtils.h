#ifndef UTILITIES_GENERAL_MUTEXUTILS_H
#define UTILITIES_GENERAL_MUTEXUTILS_H
//
//  thread Utilities...
//
#include <boost/thread/recursive_mutex.hpp>
#include <boost/thread/mutex.hpp>
#include <boost/thread/xtime.hpp>
#include <boost/thread/condition.hpp>
#include <boost/thread/tss.hpp>

#include <iosfwd>

class MutexI {
public:
  virtual ~MutexI(){}
  virtual void lock()=0;
  virtual void unlock()=0;

  class scoped_lock {
    MutexI & m;
  public:
    scoped_lock(MutexI& im) :m(im) { m.lock();}
    ~scoped_lock() { m.unlock();}
  };

};

struct  DummyMutex : public MutexI {
  void lock(){}
  void unlock(){}
};

template<class M=boost::recursive_mutex>
class AnyMutex : public MutexI {
  M & m;
public:
  typedef M Mutex;
  typedef MutexI::scoped_lock scoped_lock;
  AnyMutex(M & im) : m(im){}
  void lock() { m.lock();}
  void unlock() { m.unlock();}
};


std::ostream & TSafeOstream();


template<class M>
class lockSentry {
public:
  typedef M Mutex;

  explicit lockSentry(M& im) :m(im) {}
  ~lockSentry() {}
  
  operator typename M::scoped_lock & () { return m;} 
  typename M::scoped_lock & operator()() { return m;} 

private:
  typename M::scoped_lock m;
  
};

typedef lockSentry<boost::recursive_mutex> LockMutex;

typedef lockSentry<boost::mutex> SimpleLockMutex;

namespace Capri {

  // the global mutex...
  extern LockMutex::Mutex glMutex;

}


template<class M=boost::recursive_mutex>
struct ThreadTraits {

  typedef M Mutex;

  /// a static safe class level mutex...
  template<class T> static  Mutex & myMutex(const T*) {
    static Mutex locm;
    return locm;
  }


};


template<class T, class M=boost::recursive_mutex>
class classLock : private lockSentry<M> {
 public:
  explicit classLock(const T* me) : lockSentry<M>(ThreadTraits<M>::myMutex(me)){}
  ~classLock(){}
};


struct boostFuture {
  boost::xtime xt;

  boostFuture(int i) {
#if BOOST_VERSION >= 105000
    boost::xtime_get(&xt, boost::TIME_UTC_);
#else
    boost::xtime_get(&xt, boost::TIME_UTC);
#endif
    xt.sec += i;
  }
  
  operator const boost::xtime & () const { return xt;}
  
};

inline pthread_t thread_self_tid() { return pthread_self();}



template<typename T>
class ThreadMessage {
public:
  typedef T Case;

  const Case& operator()() const {
    SimpleLockMutex gl(lock);
    doit.wait(gl());
    return it;
  }
  
  void go(const Case& i) {
    SimpleLockMutex gl(lock);
    it=i;
    doit.notify_all(); 
  }
  
  const Case& get() const { return it;} 

private:

  mutable SimpleLockMutex::Mutex lock;
  mutable boost::condition doit;
  Case it;
};

namespace {
  template<typename T> 
  struct defF {
    T * operator()() {
      return new T();
    }
  };

}

template<typename T, typename F=defF<T> >
class ThreadSingleton {
public:

  static T & instance(F&f=factory()) {
    static boost::thread_specific_ptr<T> me;
    if (!me.get()) {
      T * t = buildMe(f);
      me.reset(t);
    }
    return *me;
  }

private:
  static F & factory() {
    static F f;
    return f;
  }

  static T * buildMe(F&f) {
    return f();
  }

};

#endif // UTILITIES_THREADS_MUTEXUTILS_H
