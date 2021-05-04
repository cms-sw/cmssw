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

    A class that inherits from OscarProducer must have a constructor that takes
a 'const edm::ParameterSet&' as its only argument.  This constructor will be
called by the dynamic loading code.
*/
//
// Original Author:
//         Created:  Tue Nov 22 15:35:11 EST 2005
//

class SimWatcher {
public:
  SimWatcher() {}
  virtual ~SimWatcher() {}

  SimWatcher(const SimWatcher &) = delete;
  const SimWatcher &operator=(const SimWatcher &) = delete;
};

#endif
