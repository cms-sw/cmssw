#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_SEAL_MODULE();

#include "RecoTracker/TkTrackingRegions/interface/TrackingRegionProducerFactory.h"
#include "RecoTracker/TkTrackingRegions/interface/TrackingRegionProducer.h"
#include "GlobalTrackingRegionProducer.h"

DEFINE_SEAL_PLUGIN(TrackingRegionProducerFactory, GlobalTrackingRegionProducer, "GlobalRegionProducer");

