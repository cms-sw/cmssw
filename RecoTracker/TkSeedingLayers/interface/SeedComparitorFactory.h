#ifndef RecoTracker_TkSeedingLayers_SeedComparitorFactory_H
#define RecoTracker_TkSeedingLayers_SeedComparitorFactory_H

#include "RecoTracker/TkSeedingLayers/interface/SeedComparitor.h"
namespace edm {class ParameterSet; class ConsumesCollector;}

#include "FWCore/PluginManager/interface/PluginFactory.h"

typedef edmplugin::PluginFactory<SeedComparitor *(const edm::ParameterSet&, edm::ConsumesCollector&)> SeedComparitorFactory;

#endif
