#ifndef HcalSimAlgos_HcalTimeSlewSim_h
#define HcalSimAlgos_HcalTimeSlewSim_h

/** Applies a correction for time slewing
    Makes bigger signals come at a delayed time

 \Author Rick Wilkinson
 */


#include "SimCalorimetry/CaloSimAlgos/interface/CaloVSimParameterMap.h"
#include "CalibFormats/CaloObjects/interface/CaloSamples.h"

namespace CLHEP {
  class HepRandomEngine;
}

class HcalTimeSlewSim
{
public:
  HcalTimeSlewSim(const CaloVSimParameterMap * parameterMap, double minFCToDelay);

  void delay(CaloSamples & samples, CLHEP::HepRandomEngine*) const;

private:
  double charge(const CaloSamples & samples) const;

  const CaloVSimParameterMap * theParameterMap;
  double minFCToDelay_;
};

#endif
