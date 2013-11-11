#ifndef RecoTracker_TkTrackingRegions_TrackingRegionProducerFactory_H
#define RecoTracker_TkTrackingRegions_TrackingRegionProducerFactory_H

#include "RecoTracker/TkTrackingRegions/interface/TrackingRegionProducer.h"
namespace edm {class ParameterSet;}

#include "FWCore/PluginManager/interface/PluginFactory.h"

typedef edmplugin::PluginFactory<TrackingRegionProducer *(const edm::ParameterSet &)> TrackingRegionProducerFactory;
#endif

