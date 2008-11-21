#ifndef HcalSimAlgos_HPDNoiseGenerator_h
#define HcalSimAlgos_HPDNoiseGenerator_h

#include "SimCalorimetry/CaloSimAlgos/interface/CaloVNoiseSignalGenerator.h"
#include <vector>
class HcalSimParameterMap;

class HPDNoiseGenerator : public CaloVNoiseSignalGenerator
{
public:
  HPDNoiseGenerator(const HcalSimParameterMap * parameterMap);

  void getNoiseSignals(std::vector<CaloSamples> & result);

private:
  void fC2pe(CaloSamples & samples) const;

  const HcalSimParameterMap * theParameterMap;
};

#endif
 
