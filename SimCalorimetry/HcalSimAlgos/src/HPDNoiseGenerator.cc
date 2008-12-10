#include "SimCalorimetry/HcalSimAlgos/interface/HPDNoiseGenerator.h"
#include "SimCalorimetry/HcalSimAlgos/interface/HcalSimParameterMap.h"

HPDNoiseGenerator::HPDNoiseGenerator(const edm::ParameterSet & pset, const HcalSimParameterMap * parameterMap)
: theLibraryReader(pset),
  theParameterMap(parameterMap)
{
}


void HPDNoiseGenerator::fillNoiseSignals()
{
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
    theNoiseSignals.push_back(newSamples);
  }
}

void HPDNoiseGenerator::fC2pe(CaloSamples & samples) const
{
  float factor = 1./theParameterMap->simParameters(samples.id()).photoelectronsToAnalog();
  samples *= factor;
}

