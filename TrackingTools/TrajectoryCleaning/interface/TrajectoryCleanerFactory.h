#ifndef TrackingTools_TrajectoryCleaning_TrajectoryCleanerFactory_H
#define TrackingTools_TrajectoryCleaning_TrajectoryCleanerFactory_H

#include "TrackingTools/TrajectoryCleaning/interface/TrajectoryCleaner.h"
#include "FWCore/PluginManager/interface/PluginFactory.h" 

typedef edmplugin::PluginFactory<TrajectoryCleaner *( const edm::ParameterSet & )> TrajectoryCleanerFactory;

#endif
