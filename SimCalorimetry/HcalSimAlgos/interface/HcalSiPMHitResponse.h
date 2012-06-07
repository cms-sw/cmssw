// -*- C++ -*- 
#ifndef HcalSimAlgos_HcalSiPMHitResponse_h
#define HcalSimAlgos_HcalSiPMHitResponse_h

#include "SimCalorimetry/CaloSimAlgos/interface/CaloHitResponse.h"

#include <map>
#include <set>

class HcalSiPM;

class CaloHitTimeAndEnergy {
public:
  CaloHitTimeAndEnergy(float theTime, float theEnergy) : myTime(theTime), myEnergy(theEnergy) {}
  float time() const {return myTime;}
  float energy() const {return myEnergy;}
private:
  float myTime;
  float myEnergy;
};

class PCaloHitCompareTimes {
public:
  bool operator()(const CaloHitTimeAndEnergy& a, 
		  const CaloHitTimeAndEnergy& b) const {
    return a.time()<b.time();
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
  typedef std::multiset <CaloHitTimeAndEnergy, PCaloHitCompareTimes> SortedHitSet;

  virtual CaloSamples makeSiPMSignal(const DetId& id, const CaloHitTimeAndEnergy & hit, int & integral) const;

  HcalSiPM * theSiPM;
  double theRecoveryTime;

  std::map< DetId, SortedHitSet > sortedhits;
};

#endif //HcalSimAlgos_HcalSiPMHitResponse_h
