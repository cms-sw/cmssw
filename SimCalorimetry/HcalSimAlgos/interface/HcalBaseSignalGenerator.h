#ifndef HcalSimAlgos_HcalBaseSignalGenerator_h
#define HcalSimAlgos_HcalBaseSignalGenerator_h

#include "SimCalorimetry/CaloSimAlgos/interface/CaloVNoiseSignalGenerator.h"
#include "SimCalorimetry/HcalSimAlgos/interface/HcalSimParameterMap.h"
class HcalElectronicsSim;

class HcalBaseSignalGenerator : public CaloVNoiseSignalGenerator {
public:
  HcalBaseSignalGenerator() : theParameterMap(nullptr), theElectronicsSim(nullptr) {}

  ~HcalBaseSignalGenerator() override {}

  void setParameterMap(HcalSimParameterMap* map) { theParameterMap = map; }

  // can be needed to set starting cap ID
  void setElectronicsSim(HcalElectronicsSim* electronicsSim) { theElectronicsSim = electronicsSim; }

protected:
  void fC2pe(CaloSamples& samples) const {
    assert(theParameterMap != nullptr);
    float factor = 1. / theParameterMap->simParameters(samples.id()).photoelectronsToAnalog(samples.id());
    samples *= factor;
  }

  HcalSimParameterMap* theParameterMap;
  HcalElectronicsSim* theElectronicsSim;
};

#endif
