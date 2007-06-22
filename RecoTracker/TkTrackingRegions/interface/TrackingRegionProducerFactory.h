#ifndef RecoTracker_TkTrackingRegions_TrackingRegionProducerFactory_H
#define RecoTracker_TkTrackingRegions_TrackingRegionProducerFactory_H

#include "RecoTracker/TkTrackingRegions/interface/TrackingRegionProducer.h"
namespace edm {class ParameterSet;}

#include "FWCore/PluginManager/interface/PluginFactory.h"

class TrackingRegionProducerFactory 
   : public seal::PluginFactory< TrackingRegionProducer * (const edm::ParameterSet&) > {
public:
  TrackingRegionProducerFactory();
  virtual ~TrackingRegionProducerFactory();
  static TrackingRegionProducerFactory * get();
};
#endif

