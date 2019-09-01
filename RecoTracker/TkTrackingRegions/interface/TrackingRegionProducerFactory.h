#ifndef RecoTracker_TkTrackingRegions_TrackingRegionProducerFactory_H
#define RecoTracker_TkTrackingRegions_TrackingRegionProducerFactory_H

#include "RecoTracker/TkTrackingRegions/interface/TrackingRegionProducer.h"
namespace edm {
  class ParameterSet;
  class ConsumesCollector;
}  // namespace edm

#include "FWCore/PluginManager/interface/PluginFactory.h"

typedef edmplugin::PluginFactory<TrackingRegionProducer *(const edm::ParameterSet &, edm::ConsumesCollector &&)>
    TrackingRegionProducerFactory;
typedef edmplugin::PluginFactory<TrackingRegionProducer *(const edm::ParameterSet &)>
    TrackingRegionProducerFactoryNoConsumes;
#endif
