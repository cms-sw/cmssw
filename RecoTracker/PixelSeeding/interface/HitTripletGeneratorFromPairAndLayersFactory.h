#ifndef RecoTracker_PixelSeeding_HitTripletGeneratorFromPairAndLayersFactory_h
#define RecoTracker_PixelSeeding_HitTripletGeneratorFromPairAndLayersFactory_h

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
