#ifndef SimG4Core_Notification_Observer_H
#define SimG4Core_Notification_Observer_H

#include <boost/signal.hpp>

template<typename T>
class Observer : public boost::signals::trackable 
{
public:
    Observer() {}
    virtual ~Observer() {}
    virtual void update(T) const = 0;
};

#endif
