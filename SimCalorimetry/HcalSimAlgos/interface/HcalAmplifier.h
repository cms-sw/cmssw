#ifndef HcalSimAlgos_HcalAmplifier_h
#define HcalSimAlgos_HcalAmplifier_h
  
#include "CalibFormats/CaloObjects/interface/CaloSamples.h"
class HcalDbService;
class HcalSimParameterMap;

class HcalAmplifier {
public:
  HcalAmplifier(const HcalSimParameterMap * parameters, bool addNoise);
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
  const HcalSimParameterMap * theParameterMap;
  unsigned theStartingCapId;
  bool addNoise_;
};

#endif
