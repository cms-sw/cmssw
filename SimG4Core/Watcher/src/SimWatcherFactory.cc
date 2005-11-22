#include "SimG4Core/Watcher/interface/SimWatcherFactory.h"

SimWatcherFactory SimWatcherFactory::s_instance;

SimWatcherFactory::SimWatcherFactory()
    : seal::PluginFactory<SimWatcherMakerBase * ()>("CMS Simulation SimWatcherFactory")
{}

SimWatcherFactory::~SimWatcherFactory() {}

SimWatcherFactory * SimWatcherFactory::get() { return & s_instance; }
