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

The Observer class has two optional template arguments, TPre and TPost.  These arguments allow you
to specify a functional object (a class that has an 'operator()(T)' method defined) which will be called
before (TPre) and after (TPost) the overrided 'update()' method is called.

*/
//
// $Id: Observer.h,v 1.6 2005/11/21 16:18:10 chrjones Exp $
//

#include <boost/signal.hpp>
#include "SimG4Core/Notification/interface/SimSlotAdapter.h"

namespace observer_helper {
   /**
   class DoNothing is used as the default helper functional object that
    will be called before and then after Observer::update.
    The compiler should be able to optimize away the function call.
   */
   template<class T>
   struct DoNothing {
      void operator()(T){ }
   };
   
   template<class T>
      DoNothing<T> makeDoNothing() {return DoNothing<T>();}

   /**
   class SlotForUpdateCaller is a functional object that adapts an
    Observer to be callable from a SimSlotAdapter
      */
   template<class T, class TObserver>
      struct SlotForUpdateCaller {
         SlotForUpdateCaller() : m_obs(0) {}
         void operator()(T iT) {
            m_obs->slotForUpdate(iT);
         }
         
         TObserver* m_obs;
      };
}

template<class T, 
         class TPre=observer_helper::DoNothing<T>, 
         class TPost=observer_helper::DoNothing<T> >
class Observer : public SimSlotAdapter<T,observer_helper::SlotForUpdateCaller<T,Observer<T,TPre,TPost> > >
{
public:
    typedef TPre PreUpdater;
    typedef TPost PostUpdater;
    typedef SimSlotAdapter<T, observer_helper::SlotForUpdateCaller<T,Observer<T,TPre,TPost> > > Base;
    
    Observer(PreUpdater pre=PreUpdater(), 
             PostUpdater post=PostUpdater() ): Base(), m_pre(pre), m_post(post) 
    { //'this' not set until reach the body of the constructor                
      this->functor().m_obs=this; }
    virtual ~Observer() {}

    /** This method is what is called when the signal is actually sent.  The signal is not sent
       directly to 'update' because if we did
       1) If the user did not declare 'update' to be 'public' then we get a compilation failure 
       2) we would not have a 'hook' to allow the 'pre' and 'post' functions to be called
       */
    void slotForUpdate(T iT) {
       m_pre(iT);
       update(iT);
       m_post(iT);
    }
protected:
    ///This routine will be called when the appropriate signal arrives
    virtual void update(T)  = 0;
private:
    PreUpdater m_pre;
    PostUpdater m_post;
};

#endif
