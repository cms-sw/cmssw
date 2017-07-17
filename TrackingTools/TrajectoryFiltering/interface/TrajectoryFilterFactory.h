#ifndef TrackingTools_TrajectoryFilter_TrajectoryFilterFactory_H
#define TrackingTools_TrajectoryFilter_TrajectoryFilterFactory_H

#include "FWCore/PluginManager/interface/PluginFactory.h"
#include "TrackingTools/TrajectoryFiltering/interface/TrajectoryFilter.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

namespace edm {
  class ConsumesCollector;
}

typedef edmplugin::PluginFactory< TrajectoryFilter* (const edm::ParameterSet&, edm::ConsumesCollector& iC) > TrajectoryFilterFactory;

#endif
