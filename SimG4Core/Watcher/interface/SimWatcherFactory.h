#ifndef SimG4Core_SimWatcherFactory_H
#define SimG4Core_SimWatcherFactory_H

#include "SimG4Core/Watcher/interface/SimWatcher.h"
#include "SimG4Core/Watcher/interface/SimWatcherMaker.h"

#include "FWCore/PluginManager/interface/PluginFactory.h"

typedef edmplugin::PluginFactory<SimWatcherMakerBase *() > SimWatcherFactory ;

//This pattern was taken from the framework factory code

#define DEFINE_SIMWATCHER(type) \
  DEFINE_EDM_PLUGIN(SimWatcherFactory, SimWatcherMaker<type>,#type)

#endif
