// -*- C++ -*- 
#ifndef HcalSimAlgos_HcalSiPMHitResponse_h
#define HcalSimAlgos_HcalSiPMHitResponse_h

#include "SimCalorimetry/CaloSimAlgos/interface/CaloHitResponse.h"
#include "SimCalorimetry/HcalSimAlgos/interface/HcalSiPMRecovery.h"

#include <map>
#include <set>

class HcalSiPM;

class PCaloHitCompareTimes {
public:
  bool operator()(const PCaloHit * a, 
		  const PCaloHit * b) const {
    return a->time()<b->time();
  }
};

class HcalSiPMHitResponse : public CaloHitResponse {

public:
  HcalSiPMHitResponse(const CaloVSimParameterMap * parameterMap, 
		      const CaloShapes * shapes);

  virtual ~HcalSiPMHitResponse();

  virtual void initializeHits();

  virtual void finalizeHits();

  virtual void add(const PCaloHit& hit);

  using CaloHitResponse::add;

  virtual void run(MixCollection<PCaloHit> & hits);

  virtual void setRandomEngine(CLHEP::HepRandomEngine & engine);

 private:
  typedef std::multiset <PCaloHit, PCaloHitCompareTimes> SortedHitSet;

  virtual CaloSamples makeSiPMSignal(const DetId& id, const PCaloHit& hit, int & integral) const;

  HcalSiPM * theSiPM;
  double theRecoveryTime;

  std::map< DetId, HcalSiPMRecovery > pixelHistory;
};

#endif //HcalSimAlgos_HcalSiPMHitResponse_h
