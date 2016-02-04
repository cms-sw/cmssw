#ifndef UTILITIES_TIMING_BASEEVENT_H
#define UTILITIES_TIMING_BASEEVENT_H

template<class T>
class BaseEvent {
public:
  typedef T event;

  virtual ~BaseEvent(){}

  virtual void operator()(const T&)=0;


};

#endif // UTILITIES_TIMING_BASEEVENT_H
