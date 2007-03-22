#include "RecoTracker/TkTrackingRegions/interface/TrackingRegionProducerFactory.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include <iostream>

TrackingRegionProducerFactory::TrackingRegionProducerFactory()
  : seal::PluginFactory<TrackingRegionProducer*(const edm::ParameterSet & p)>("TrackingRegionProducerFactory")
{ }

TrackingRegionProducerFactory::~TrackingRegionProducerFactory()
{ }

TrackingRegionProducerFactory * TrackingRegionProducerFactory::get()
{
  static TrackingRegionProducerFactory theTrackingRegionProducerFactory;
  return & theTrackingRegionProducerFactory;
}

