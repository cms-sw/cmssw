#ifndef Notification_SimSlotAdapter_h
#define Notification_SimSlotAdapter_h
// -*- C++ -*-
//
// Package:     Notification
// Class  :     SimSlotAdapter
// 
/**\class SimSlotAdapter SimSlotAdapter.h SimG4Core/Notification/interface/SimSlotAdapter.h

 Description: Adapts a 'functional object' to work with the signals emitted by the OscarProducer

 Usage:
    When the OscarProducer loads an object from its configuration file, it looks to see if that
    object inherits from SimSlotAdapter and if so, the OscarProducer will send the appropriate 
    signals (e.g., BeginOfTrack) to that object.

    The template argument 'TFunc' is the class of the functional object being adapted.  
    The 'T' argument is the actual argument of the signal being sent.

*/
//
// Original Author:  Chris Jones
//         Created:  Sat Nov 19 18:57:13 EST 2005
// $Id$
//

// system include files

// user include files
#include <boost/signal.hpp>

// forward declarations

template<class T, class TFunc>
class SimSlotAdapter : public boost::signals::trackable
{

   public:
      SimSlotAdapter(TFunc iFunc=TFunc()): m_func(iFunc) {}
      virtual ~SimSlotAdapter() {}

      // ---------- member functions ---------------------------
      void slot(T iT) {
         m_func(iT) ;
      }
   
   protected:
      TFunc& functor() { return m_func; }
   private:
      //SimSlotAdapter(const SimSlotAdapter&); // stop default

      //const SimSlotAdapter& operator=(const SimSlotAdapter&); // stop default

      // ---------- member data --------------------------------
      TFunc m_func;
};


#endif
