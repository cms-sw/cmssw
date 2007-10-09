#ifndef CastorSim_CastorAmplifier_h
#define CastorSim_CastorAmplifier_h
  
#include "CalibFormats/CaloObjects/interface/CaloSamples.h"
#include "SimCalorimetry/CaloSimAlgos/interface/CaloVSimParameterMap.h"
#include "CLHEP/Random/RandGaussQ.h"

class HcalDbService;

class CastorAmplifier {
public:
  CastorAmplifier(const CaloVSimParameterMap * parameters, bool addNoise);
  virtual ~CastorAmplifier(){ delete theRandGaussQ; }

  /// the Producer will probably update this every event
  void setDbService(const HcalDbService * service) {
    theDbService = service;
   }

  void setRandomEngine(CLHEP::HepRandomEngine & engine);

  virtual void amplify(CaloSamples & linearFrame) const;

  void setStartingCapId(int capId) {theStartingCapId = capId;}

private:
  const HcalDbService * theDbService;
  CLHEP::RandGaussQ * theRandGaussQ;
  const CaloVSimParameterMap * theParameterMap;

  unsigned theStartingCapId;
  bool addNoise_;
};

#endif
