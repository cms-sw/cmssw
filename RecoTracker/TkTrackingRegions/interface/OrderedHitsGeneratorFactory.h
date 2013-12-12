#ifndef RecoTracker_TkTrackingRegions_OrderedHitsGeneratorFactory_H
#define RecoTracker_TkTrackingRegions_OrderedHitsGeneratorFactory_H

#include "RecoTracker/TkTrackingRegions/interface/OrderedHitsGenerator.h"
namespace edm {class ParameterSet;}

#include "FWCore/PluginManager/interface/PluginFactory.h"

typedef edmplugin::PluginFactory<OrderedHitsGenerator *(const edm::ParameterSet &)> OrderedHitsGeneratorFactory;

#endif

