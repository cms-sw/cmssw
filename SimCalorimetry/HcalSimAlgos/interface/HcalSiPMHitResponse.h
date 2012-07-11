// -*- C++ -*- 
#ifndef HcalSimAlgos_HcalSiPMHitResponse_h
#define HcalSimAlgos_HcalSiPMHitResponse_h

#include "SimCalorimetry/CaloSimAlgos/interface/CaloHitResponse.h"

class HcalSiPM;

class HcalSiPMHitResponse : public CaloHitResponse {

public:
  HcalSiPMHitResponse(const CaloVSimParameterMap * parameterMap, 
		      const CaloShapes * shapes);

  virtual ~HcalSiPMHitResponse();

  virtual void run(MixCollection<PCaloHit> & hits);

  virtual void setRandomEngine(CLHEP::HepRandomEngine & engine);

 protected:

  virtual CaloSamples makeSiPMSignal(const PCaloHit & inHit, int & integral) const;

  HcalSiPM * theSiPM;
  double theRecoveryTime;
};

#endif //HcalSimAlgos_HcalSiPMHitResponse_h
