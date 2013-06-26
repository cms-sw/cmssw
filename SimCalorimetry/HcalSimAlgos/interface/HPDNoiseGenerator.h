#ifndef HcalSimAlgos_HPDNoiseGenerator_h
#define HcalSimAlgos_HPDNoiseGenerator_h

#include "SimCalorimetry/HcalSimAlgos/interface/HcalBaseSignalGenerator.h"
#include "SimCalorimetry/HcalSimAlgos/interface/HPDNoiseLibraryReader.h"
#include "FWCore/ParameterSet/interface/ParameterSetfwd.h"
#include <vector>
class HcalSimParameterMap;

class HPDNoiseGenerator : public HcalBaseSignalGenerator
{
public:
  HPDNoiseGenerator(const edm::ParameterSet & pset);
  virtual ~HPDNoiseGenerator() {}

  void fillNoiseSignals();

private:
  HPDNoiseLibraryReader theLibraryReader;
};

#endif
 
