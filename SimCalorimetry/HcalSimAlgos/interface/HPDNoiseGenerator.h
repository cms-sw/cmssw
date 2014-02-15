#ifndef HcalSimAlgos_HPDNoiseGenerator_h
#define HcalSimAlgos_HPDNoiseGenerator_h

#include "SimCalorimetry/HcalSimAlgos/interface/HcalBaseSignalGenerator.h"
#include "SimCalorimetry/HcalSimAlgos/interface/HPDNoiseLibraryReader.h"
#include "FWCore/ParameterSet/interface/ParameterSetfwd.h"
#include <vector>
class HcalSimParameterMap;

namespace CLHEP {
  class HepRandomEngine;
}

class HPDNoiseGenerator : public HcalBaseSignalGenerator
{
public:
  HPDNoiseGenerator(const edm::ParameterSet & pset);
  virtual ~HPDNoiseGenerator() {}

  void fillNoiseSignals(CLHEP::HepRandomEngine*) override;

private:
  HPDNoiseLibraryReader theLibraryReader;
};

#endif
 
