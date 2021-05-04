#ifndef EcalSimAlgos_EcalBaseSignalGenerator_h
#define EcalSimAlgos_EcalBaseSignalGenerator_h

#include "SimCalorimetry/CaloSimAlgos/interface/CaloVNoiseSignalGenerator.h"
#include "SimCalorimetry/EcalSimAlgos/interface/EcalSimParameterMap.h"

class EcalBaseSignalGenerator : public CaloVNoiseSignalGenerator {
public:
  EcalBaseSignalGenerator() {}

  ~EcalBaseSignalGenerator() override {}

protected:
  //  void fC2pe(CaloSamples & samples) const
  // {
  //  assert(theParameterMap != 0);
  //  float factor = 1./theParameterMap->simParameters(samples.id()).photoelectronsToAnalog(samples.id());
  //  samples *= factor;
  // }
};

#endif
