#include "SimCalorimetry/HcalSimAlgos/interface/HPDNoiseGenerator.h"

HPDNoiseGenerator::HPDNoiseGenerator(const edm::ParameterSet & pset)
: HcalBaseSignalGenerator(),
  theLibraryReader(pset)
{
}


void HPDNoiseGenerator::fillNoiseSignals()
{
  theNoiseSignals.clear();
  std::vector<std::pair <HcalDetId, const float* > > noise = theLibraryReader.getNoisyHcalDetIds();
  for(std::vector<std::pair <HcalDetId, const float* > >::const_iterator noiseItr = noise.begin();
      noiseItr != noise.end(); ++noiseItr)
  {
    CaloSamples newSamples(noiseItr->first, 10);
    for(unsigned i = 0; i < 10; ++i)
    {
      newSamples[i] = (noiseItr->second)[i];
    }
    // result should come back in units of photoelectrons
    fC2pe(newSamples);
    theNoiseSignals.push_back(std::move(newSamples));
  }
}

