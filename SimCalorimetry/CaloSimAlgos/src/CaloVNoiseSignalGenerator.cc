#include "SimCalorimetry/CaloSimAlgos/interface/CaloVNoiseSignalGenerator.h"
#include "FWCore/Utilities/interface/Algorithms.h"
#include <iostream>

CaloVNoiseSignalGenerator::CaloVNoiseSignalGenerator()
: theNoiseSignals(),
  theDetIds()
{
}


void CaloVNoiseSignalGenerator::fillEvent()
{
  theDetIds.clear();
  fillNoiseSignals();
  fillDetIds();
}

void CaloVNoiseSignalGenerator::setNoiseSignals(const std::vector<CaloSamples> & noiseSignals)
{
  theNoiseSignals = noiseSignals;
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

    //    std::cout << "Noise DetId " << sampleItr->id().rawId() << std::endl;

  }
  edm::sort_all(theDetIds);
}



