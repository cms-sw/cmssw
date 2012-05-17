// -*- C++ -*- 
#ifndef HcalSimAlgos_HcalSiPMHitResponse_h
#define HcalSimAlgos_HcalSiPMHitResponse_h

#include "SimCalorimetry/CaloSimAlgos/interface/CaloHitResponse.h"
#include "SimCalorimetry/HcalSimAlgos/src/HcalTDCParameters.h"
class HcalSiPM;

class HcalSiPMHitResponse : public CaloHitResponse {

public:
  HcalSiPMHitResponse(const CaloVSimParameterMap * parameterMap, 
		      const CaloVShape * shape, const CaloVShape * integratedShape);

  virtual ~HcalSiPMHitResponse();

  virtual void run(MixCollection<PCaloHit> & hits);

  virtual void setRandomEngine(CLHEP::HepRandomEngine & engine);

  virtual CaloSamples makeBlankSignal(const DetId & detId) const;
 protected:

  virtual CaloSamples makeSiPMSignal(const PCaloHit & inHit, int & integral) const;
  HcalTDCParameters theTDCParameters;
  const CaloVShape * theIntegratedShape;
  HcalSiPM * theSiPM;
  double theRecoveryTime;
  int theShapeNormalization;
};

#endif //HcalSimAlgos_HcalSiPMHitResponse_h
