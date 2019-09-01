#ifndef Watcher_SimWatcherMakerBase_h
#define Watcher_SimWatcherMakerBase_h
// -*- C++ -*-
//
// Package:     Watcher
// Class  :     SimWatcherMakerBase
//
/**\class SimWatcherMakerBase SimWatcherMakerBase.h
 SimG4Core/Watcher/interface/SimWatcherMakerBase.h

 Description: Base class for the 'maker' which creates Watchers

 Usage:
    This class is the interface for creating a Watcher and for connnecting
 the appropriate OSCAR signals to that Watcher

*/
//
// Original Author:  Chris D Jones
//         Created:  Tue Nov 22 13:03:39 EST 2005
//

// system include files
#include <memory>

// user include files

// forward declarations
class SimActivityRegistry;
namespace edm {
  class ParameterSet;
}
class SimWatcher;
class SimProducer;
class SimWatcherMakerBase {
public:
  SimWatcherMakerBase() {}
  virtual ~SimWatcherMakerBase() {}

  // ---------- const member functions ---------------------
  virtual void make(const edm::ParameterSet &,
                    SimActivityRegistry &,
                    std::shared_ptr<SimWatcher> &,
                    std::shared_ptr<SimProducer> &) const = 0;
};

#endif
