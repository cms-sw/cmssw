// -*- C++ -*- 
#ifndef HcalSimAlgos_HcalSiPMHitResponse_h
#define HcalSimAlgos_HcalSiPMHitResponse_h

#include "SimCalorimetry/CaloSimAlgos/interface/CaloHitResponse.h"
#include "SimCalorimetry/HcalSimAlgos/interface/HcalSiPMRecovery.h"
#include "SimCalorimetry/HcalSimAlgos/interface/HcalTDCParameters.h"

#include <map>
#include <set>
#include <vector>

#include "CLHEP/Random/RandFlat.h"

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

  typedef std::vector<unsigned int> photonTimeHist;
  typedef std::map< DetId, photonTimeHist > photonTimeMap;

  virtual void initializeHits();

  virtual void finalizeHits();

  virtual void add(const PCaloHit& hit);

  virtual void add(const CaloSamples& signal);

  virtual void run(MixCollection<PCaloHit> & hits);

  virtual void setRandomEngine(CLHEP::HepRandomEngine & engine);

  virtual CaloSamples makeBlankSignal(const DetId & detId) const;

  static double Y11TimePDF( double t );

 private:
  typedef std::multiset <PCaloHit, PCaloHitCompareTimes> SortedHitSet;

  virtual CaloSamples makeSiPMSignal(const DetId& id, const PCaloHit& hit, int & integral) const;
  virtual CaloSamples makeSiPMSignal(DetId const& id, photonTimeHist const& photons) const;

  double generatePhotonTime() const;

  HcalSiPM * theSiPM;
  double theRecoveryTime;
  int const TIMEMULT;
  float const Y11RANGE;
  float const Y11MAX;
  float const Y11TIMETORISE;

  photonTimeMap precisionTimedPhotons;
  HcalTDCParameters theTDCParams;

  CLHEP::RandFlat * theRndFlat;

};

#endif //HcalSimAlgos_HcalSiPMHitResponse_h
