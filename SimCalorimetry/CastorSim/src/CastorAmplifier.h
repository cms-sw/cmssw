#ifndef CastorSim_CastorAmplifier_h
#define CastorSim_CastorAmplifier_h

#include "CalibFormats/CaloObjects/interface/CaloSamples.h"
#include "SimCalorimetry/CastorSim/src/CastorSimParameterMap.h"

class CastorDbService;

namespace CLHEP {
  class HepRandomEngine;
}

class CastorAmplifier {
public:
  CastorAmplifier(const CastorSimParameterMap *parameters, bool addNoise);
  virtual ~CastorAmplifier() {}

  /// the Producer will probably update this every event
  void setDbService(const CastorDbService *service) { theDbService = service; }

  virtual void amplify(CaloSamples &linearFrame, CLHEP::HepRandomEngine *) const;

  void setStartingCapId(int capId) { theStartingCapId = capId; }

private:
  const CastorDbService *theDbService;
  const CastorSimParameterMap *theParameterMap;

  unsigned theStartingCapId;
  bool addNoise_;
};

#endif
