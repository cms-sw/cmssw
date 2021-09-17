#ifndef SimG4Core_Notification_Signaler_h
#define SimG4Core_Notification_Signaler_h
// -*- C++ -*-
//
// Package:     Notification
// Class  :     Signaler
//
/**\class Signaler Signaler.h SimG4Core/Notification/interface/Signaler.h

 Description: Manages a particular signal of the SimActivityRegistry

 Usage:
    This is an internal detail of how the SimActivityRegistry does its job.

    Having Signaler inherit frmo Observer<const T*> allows one Signaler to forward its signal
    to another Signaler.

    All connected Observers are required to have a lifetime greater than the Signaler to which they are 
    attached.
*/
//
// Original Author:
//         Created:  Sat Dec  1 10:17:20 EST 2007
//

// system include files
#include <vector>

// user include files
#include "SimG4Core/Notification/interface/Observer.h"

// forward declarations

namespace sim_act {
  template <typename T>
  class Signaler : public Observer<const T*> {
  public:
    typedef Observer<const T*>* slot_type;
    Signaler() {}
    ~Signaler() override {}

    // ---------- const member functions ---------------------
    void operator()(const T* iSignal) const {
      typedef typename std::vector<Observer<const T*>*>::const_iterator iterator;
      for (iterator it = observers_.begin(); it != observers_.end(); ++it) {
        (*it)->slotForUpdate(iSignal);
      }
    }

    // ---------- static member functions --------------------

    // ---------- member functions ---------------------------

    ///does not take ownership of memory
    void connect(Observer<const T*>* iObs) { observers_.push_back(iObs); }

    ///does not take ownership of memory
    void connect(Observer<const T*>& iObs) { observers_.push_back(&iObs); }

    Signaler(const Signaler&) = delete;                   // stop default
    const Signaler& operator=(const Signaler&) = delete;  // stop default

  private:
    void update(const T* iData) override { this->operator()(iData); }
    // ---------- member data --------------------------------
    std::vector<Observer<const T*>*> observers_;
  };

}  // namespace sim_act
#endif
