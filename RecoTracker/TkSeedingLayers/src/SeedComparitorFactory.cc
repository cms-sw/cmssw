#include "RecoTracker/TkSeedingLayers/interface/SeedComparitorFactory.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

SeedComparitorFactory::SeedComparitorFactory()
  : seal::PluginFactory<SeedComparitor* (const edm::ParameterSet & p)>(
    "SeedComparitorFactory")
{ }

SeedComparitorFactory::~SeedComparitorFactory()
{ }

SeedComparitorFactory * SeedComparitorFactory::get()
{
  static SeedComparitorFactory theSeedComparitorFactory;
  return & theSeedComparitorFactory;
}

