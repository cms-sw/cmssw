#ifndef CastorSim_CastorAmplifier_h
#define CastorSim_CastorAmplifier_h
  
#include "CalibFormats/CaloObjects/interface/CaloSamples.h"
#include "SimCalorimetry/CastorSim/src/CastorSimParameterMap.h"
#include "CLHEP/Random/RandGaussQ.h"

class CastorDbService;

class CastorAmplifier {
public:
  CastorAmplifier(const CastorSimParameterMap * parameters, bool addNoise);
  virtual ~CastorAmplifier(){ delete theRandGaussQ; }

  /// the Producer will probably update this every event
  void setDbService(const CastorDbService * service) {
    theDbService = service;
   }

  void setRandomEngine(CLHEP::HepRandomEngine & engine);

  virtual void amplify(CaloSamples & linearFrame) const;

  void setStartingCapId(int capId) {theStartingCapId = capId;}

private:
  const CastorDbService * theDbService;
  CLHEP::RandGaussQ * theRandGaussQ;
  const CastorSimParameterMap * theParameterMap;

  unsigned theStartingCapId;
  bool addNoise_;
};

#endif
