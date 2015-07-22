#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "RecoTracker/TkTrackingRegions/interface/TrackingRegionProducerFactory.h" 	 
#include "RecoTracker/TkTrackingRegions/interface/TrackingRegionProducer.h" 	 
#include "RecoTracker/TkTrackingRegions/src/PointSeededTrackingRegionsProducer.h"
#include "RecoTracker/MeasurementDet/interface/MeasurementTrackerEvent.h"

DEFINE_EDM_PLUGIN(TrackingRegionProducerFactory, PointSeededTrackingRegionsProducer, "PointSeededTrackingRegionsProducer");
//
