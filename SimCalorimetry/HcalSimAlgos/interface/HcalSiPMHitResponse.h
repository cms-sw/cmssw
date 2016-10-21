// -*- C++ -*- 
#ifndef HcalSimAlgos_HcalSiPMHitResponse_h
#define HcalSimAlgos_HcalSiPMHitResponse_h

#include "SimCalorimetry/CaloSimAlgos/interface/CaloHitResponse.h"
#include "SimCalorimetry/HcalSimAlgos/interface/HcalSiPMRecovery.h"
#include "SimCalorimetry/HcalSimAlgos/interface/HcalTDCParameters.h"

#include <map>
#include <set>
#include <vector>

class HcalSiPM;

namespace CLHEP {
  class HepRandomEngine;
}

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
		      const CaloShapes * shapes, bool PreMix1 = false);

  virtual ~HcalSiPMHitResponse();

  typedef std::vector<unsigned int> photonTimeHist;
  typedef std::map< DetId, photonTimeHist > photonTimeMap;

  virtual void initializeHits();

  virtual void finalizeHits(CLHEP::HepRandomEngine*) override;

  virtual void add(const PCaloHit& hit, CLHEP::HepRandomEngine*) override;

  virtual void add(const CaloSamples& signal);

  virtual void addPEnoise(CLHEP::HepRandomEngine* engine);

  virtual CaloSamples makeBlankSignal(const DetId & detId) const;

  virtual void setDetIds(const std::vector<DetId> & detIds);

  static double Y11TimePDF( double t );

  double generatePhotonTime(CLHEP::HepRandomEngine*) const;

protected:
  typedef std::multiset <PCaloHit, PCaloHitCompareTimes> SortedHitSet;

  virtual CaloSamples makeSiPMSignal(DetId const& id, photonTimeHist const& photons, CLHEP::HepRandomEngine*) const;

  virtual void differentiatePreciseSamples(CaloSamples& samples, 
					   double diffNorm = 1.0) const;

private:
  HcalSiPM * theSiPM;
  double theRecoveryTime;
  int const TIMEMULT;
  float const Y11RANGE;
  float const Y11MAX;
  float const Y11TIMETORISE;
  bool PreMixDigis;

  photonTimeMap precisionTimedPhotons;
  HcalTDCParameters theTDCParams;

  const std::vector<DetId>* theDetIds;
};

#endif //HcalSimAlgos_HcalSiPMHitResponse_h
