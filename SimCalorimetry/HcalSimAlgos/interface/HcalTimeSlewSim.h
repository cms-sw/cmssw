#ifndef HcalSimAlgos_HcalTimeSlewSim_h
#define HcalSimAlgos_HcalTimeSlewSim_h

/** Applies a correction for time slewing
    Makes bigger signals come at a delayed time

 \Author Rick Wilkinson
 */


#include "SimCalorimetry/CaloSimAlgos/interface/CaloVSimParameterMap.h"
#include "CalibFormats/CaloObjects/interface/CaloSamples.h"
#include "CLHEP/Random/RandGaussQ.h"

class HcalTimeSlewSim
{
public:
  HcalTimeSlewSim(const CaloVSimParameterMap * parameterMap);

  void delay(CaloSamples & samples) const;

  void setRandomEngine(CLHEP::HepRandomEngine & engine);

private:
  double charge(const CaloSamples & samples) const;

  const CaloVSimParameterMap * theParameterMap;
  CLHEP::RandGaussQ* theRandGaussQ;
};

#endif

