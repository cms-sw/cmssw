#ifndef RecoTracker_TkSeedingLayers_SeedComparitorFactory_H
#define RecoTracker_TkSeedingLayers_SeedComparitorFactory_H

#include "RecoTracker/TkSeedingLayers/interface/SeedComparitor.h"
namespace edm {class ParameterSet;}

#include "FWCore/PluginManager/interface/PluginFactory.h"

class SeedComparitorFactory
   : public seal::PluginFactory< SeedComparitor * (const edm::ParameterSet&) > {
public:
  SeedComparitorFactory();
  virtual ~SeedComparitorFactory();
  static SeedComparitorFactory * get();
};

#endif
