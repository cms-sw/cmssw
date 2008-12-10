#include "SimCalorimetry/CaloSimAlgos/interface/CaloVNoiseSignalGenerator.h"
#include "FWCore/Utilities/interface/Algorithms.h"


CaloVNoiseSignalGenerator::CaloVNoiseSignalGenerator()
: theNoiseSignals(),
  theDetIds()
{
}


void CaloVNoiseSignalGenerator::fillEvent()
{
  theNoiseSignals.clear();
  theDetIds.clear();
  fillNoiseSignals();
  fillDetIds();
}


bool CaloVNoiseSignalGenerator::contains(const DetId & detId) const
{
  return edm::binary_search_all(theDetIds, detId.rawId());
}


void CaloVNoiseSignalGenerator::fillDetIds()
{
  theDetIds.reserve(theNoiseSignals.size());
  for(std::vector<CaloSamples>::const_iterator sampleItr = theNoiseSignals.begin();
      sampleItr != theNoiseSignals.end(); ++sampleItr)
  {
    theDetIds.push_back(sampleItr->id().rawId());
  }
  edm::sort_all(theDetIds);
}


