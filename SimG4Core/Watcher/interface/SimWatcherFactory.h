#ifndef SimG4Core_SimWatcherFactory_H
#define SimG4Core_SimWatcherFactory_H

#include "SimG4Core/Watcher/interface/SimWatcher.h"
#include "SimG4Core/Watcher/interface/SimWatcherMaker.h"

#include "PluginManager/PluginFactory.h"

class SimWatcherFactory 
    : public seal::PluginFactory<
    SimWatcherMakerBase *() >
{
public:
    virtual ~SimWatcherFactory();
    static SimWatcherFactory * get(); 
private:
    static SimWatcherFactory s_instance;
    SimWatcherFactory();
};

#define DEFINE_SIMWATCHER(type) \
  DEFINE_SEAL_PLUGIN(SimWatcherFactory, SimWatcherMaker<type>,#type);

#endif
