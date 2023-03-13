#ifndef PixelTriplets_HitTripletGeneratorFromPairAndLayersFactory_H
#define PixelTriplets_HitTripletGeneratorFromPairAndLayersFactory_H

#include "RecoTracker/PixelSeeding/interface/HitTripletGeneratorFromPairAndLayers.h"
#include "FWCore/PluginManager/interface/PluginFactory.h"

namespace edm {
  class ParameterSet;
  class ConsumesCollector;
}  // namespace edm

typedef edmplugin::PluginFactory<HitTripletGeneratorFromPairAndLayers *(const edm::ParameterSet &,
                                                                        edm::ConsumesCollector &)>
    HitTripletGeneratorFromPairAndLayersFactory;

#endif
