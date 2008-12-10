#ifndef HcalSimAlgos_HPDNoiseGenerator_h
#define HcalSimAlgos_HPDNoiseGenerator_h

#include "SimCalorimetry/CaloSimAlgos/interface/CaloVNoiseSignalGenerator.h"
#include "SimCalorimetry/HcalSimAlgos/interface/HPDNoiseLibraryReader.h"
#include "FWCore/ParameterSet/interface/ParameterSetfwd.h"
#include <vector>
class HcalSimParameterMap;

class HPDNoiseGenerator : public CaloVNoiseSignalGenerator
{
public:
  HPDNoiseGenerator(const edm::ParameterSet & pset, const HcalSimParameterMap * parameterMap);
  virtual ~HPDNoiseGenerator() {}

  void fillNoiseSignals();

private:
  void fC2pe(CaloSamples & samples) const;
  HPDNoiseLibraryReader theLibraryReader;
  const HcalSimParameterMap * theParameterMap;
};

#endif
 
