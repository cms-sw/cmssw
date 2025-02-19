#ifndef SimG4Core_Notification_Observer_H
#define SimG4Core_Notification_Observer_H
// -*- C++ -*-
//
// Package:     Notification
// Class  :     Observer
// 
/**\class Observer Observer.h SimG4Core/Notification/interface/Observer.h

Description: Adapts the COBRA signal handling for use in the OscarProducer

Usage:
If a class that is loaded by OscarProducer, via its configuraiton file, inherits from Observer<T>,
(e.g., Observer<const BeginOfTrack*>), the OscarProducer will make sure the appropriate signal
is sent to an object of that class (e.g., the object will see the BeginOfTrack signal).  To handle
the signal the class must override the 'void update(T)' method (e.g., void update(const BeginOfTrack*)).

*/
//
// $Id: Observer.h,v 1.8 2007/12/02 05:17:47 chrjones Exp $
//


template<class T>
class Observer
{
public:
    Observer() {}
    virtual ~Observer() {}

    /** This method is what is called when the signal is actually sent.  The signal is not sent
       directly to 'update' because if we did
       1) If the user did not declare 'update' to be 'public' then we get a compilation failure 
       2) we would not have a 'hook' to allow the 'pre' and 'post' functions to be called
       */
    void slotForUpdate(T iT) {
       update(iT);
    }
protected:
    ///This routine will be called when the appropriate signal arrives
    virtual void update(T)  = 0;
private:
};

#endif
