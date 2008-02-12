#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_SEAL_MODULE();

#include "RecoTracker/TkTrackingRegions/interface/TrackingRegionProducerFactory.h"
#include "RecoTracker/TkTrackingRegions/interface/TrackingRegionProducer.h"
#include "GlobalTrackingRegionWithVerticesProducer.h"
#include "GlobalTrackingRegionProducer.h"
#include "GlobalTrackingRegionProducerFromBeamSpot.h"

DEFINE_EDM_PLUGIN(TrackingRegionProducerFactory, GlobalTrackingRegionProducer, "GlobalRegionProducer");
DEFINE_EDM_PLUGIN(TrackingRegionProducerFactory, GlobalTrackingRegionProducerFromBeamSpot, "GlobalRegionProducerFromBeamSpot");
DEFINE_EDM_PLUGIN(TrackingRegionProducerFactory, GlobalTrackingRegionWithVerticesProducer, "GlobalTrackingRegionWithVerticesProducer");

