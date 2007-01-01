#ifndef HcalSimAlgos_HcalAmplifier_h
#define HcalSimAlgos_HcalAmplifier_h
  
#include "CalibFormats/CaloObjects/interface/CaloSamples.h"
#include "SimCalorimetry/CaloSimAlgos/interface/CaloVSimParameterMap.h"
class HcalDbService;

class HcalAmplifier {
public:
  HcalAmplifier(const CaloVSimParameterMap * parameters, bool addNoise);
  /// doesn't delete the pointer
  virtual ~HcalAmplifier(){}

  /// the Producer will probably update this every event
  void setDbService(const HcalDbService * service) {
    theDbService = service;
   }

  virtual void amplify(CaloSamples & linearFrame) const;

  void setStartingCapId(int capId) {theStartingCapId = capId;}

private:
  const HcalDbService * theDbService;
  const CaloVSimParameterMap * theParameterMap;
  unsigned theStartingCapId;
  bool addNoise_;
};

#endif
