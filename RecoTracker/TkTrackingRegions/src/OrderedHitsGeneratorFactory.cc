#include "RecoTracker/TkTrackingRegions/interface/OrderedHitsGeneratorFactory.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include <iostream>

OrderedHitsGeneratorFactory::OrderedHitsGeneratorFactory()
  : seal::PluginFactory<OrderedHitsGenerator* (const edm::ParameterSet & p)>("OrderedHitsGeneratorFactory")
{ }

OrderedHitsGeneratorFactory::~OrderedHitsGeneratorFactory()
{ }

OrderedHitsGeneratorFactory * OrderedHitsGeneratorFactory::get()
{
  static OrderedHitsGeneratorFactory theOrderedHitsGeneratorFactory;
  return & theOrderedHitsGeneratorFactory;
}
