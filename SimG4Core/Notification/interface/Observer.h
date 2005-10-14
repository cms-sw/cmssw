#ifndef SimG4Core_Notification_Observer_H
#define SimG4Core_Notification_Observer_H

#include <iostream>
#include <string>
#include <boost/signal.hpp>
#include <boost/bind.hpp>

template<typename Event>
class Observer : public boost::signals::trackable 
{
public:
    typedef Observer<Event> self;
    Observer() { boost::bind(&self::callBack,this,_1); init(); }
    virtual ~Observer() { end(); }
    void callBack(Event const * ev) const 
    { const_cast<self*>(this)->update(ev); }
    virtual void init() = 0;
    virtual void end() = 0;
    virtual void update(Event const * ev) const = 0;
};

#endif
