#include "SimCalorimetry/HcalSimAlgos/interface/HPDNoiseGenerator.h"
#include "SimCalorimetry/HcalSimAlgos/interface/HcalSimParameterMap.h"

HPDNoiseGenerator::HPDNoiseGenerator(const HcalSimParameterMap * parameterMap)
: theParameterMap(parameterMap)
{
}


void HPDNoiseGenerator::getNoiseSignals(std::vector<CaloSamples> & result)
{
  // result should come back in units of photoelectrons
}

void HPDNoiseGenerator::fC2pe(CaloSamples & samples) const
{
  float factor = 1./theParameterMap->simParameters(samples.id()).photoelectronsToAnalog();
  samples *= factor;
}

