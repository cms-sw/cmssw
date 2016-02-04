#ifndef RecoTracker_TkSeedGenerator_SeedCreatorFactory_H
#define RecoTracker_TkSeedGenerator_SeedCreatorFactory_H

#include "RecoTracker/TkSeedGenerator/interface/SeedCreator.h"
#include "FWCore/PluginManager/interface/PluginFactory.h"

namespace edm {class ParameterSet;}

typedef edmplugin::PluginFactory<SeedCreator *(const edm::ParameterSet&)> SeedCreatorFactory;

#endif
