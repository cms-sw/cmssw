#ifndef Watcher_SimWatcherMaker_h
#define Watcher_SimWatcherMaker_h
// -*- C++ -*-
//
// Package:     Watcher
// Class  :     SimWatcherMaker
//
/**\class SimWatcherMaker SimWatcherMaker.h
 SimG4Core/Watcher/interface/SimWatcherMaker.h

 Description: Makes a particular type of SimWatcher

 Usage:
    <usage>

*/
//
// Original Author:
//         Created:  Tue Nov 22 13:03:44 EST 2005
//

// system include files
#include <memory>

// user include files
#include "SimG4Core/Notification/interface/SimActivityRegistryEnroller.h"
#include "SimG4Core/Watcher/interface/SimWatcherMakerBase.h"
#include "SimG4Core/Watcher/interface/SimWatcher.h"
#include "SimG4Core/Watcher/interface/SimProducer.h"

// forward declarations

template <class T>
class SimWatcherMaker : public SimWatcherMakerBase {
public:
  SimWatcherMaker() {}

  // ---------- const member functions ---------------------
  void makeWatcher(const edm::ParameterSet& p,
                   SimActivityRegistry& reg,
                   SimWatcher* oWatcher,
                   SimProducer* oProd) const override {
    oWatcher = new T(p);
    SimActivityRegistryEnroller::enroll(reg, oWatcher);
    oProd = static_cast<SimProducer*>(oWatcher);
  }
};

#endif
