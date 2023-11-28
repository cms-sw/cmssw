#ifndef Watcher_SimWatcher_h
#define Watcher_SimWatcher_h
// -*- C++ -*-
//
// Package:     Watcher
// Class  :     SimWatcher
//
/**\class SimWatcher SimWatcher.h SimG4Core/Watcher/interface/SimWatcher.h

 Description: Base class for classes that 'watch' what OscarProducer does
internally

 Usage:
    By itself, this class actually does nothing except allow dynamic loading
into the OscarProducer.  To do useful work, one must inherit from this class
and one or more 'Observer<T>' classes.

*/
//
// Original Author:
//         Created:  Tue Nov 22 15:35:11 EST 2005
//

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"

class SimWatcher {
public:
  SimWatcher() {}
  virtual ~SimWatcher() {}

  // Two methods are needed to be implemented in the thread
  // safe watchers and producers
  virtual void registerConsumes(edm::ConsumesCollector){};
  virtual void beginRun(edm::EventSetup const &){};

  bool isMT() const { return applicableForMT; }

  SimWatcher(const SimWatcher &) = delete;
  const SimWatcher &operator=(const SimWatcher &) = delete;

protected:
  // Set "true" for thread safe watchers/producers
  void setMT(bool val) { applicableForMT = val; }

private:
  bool applicableForMT{false};
};

#endif
