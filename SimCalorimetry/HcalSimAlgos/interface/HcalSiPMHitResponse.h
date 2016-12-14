// -*- C++ -*- 
#ifndef HcalSimAlgos_HcalSiPMHitResponse_h
#define HcalSimAlgos_HcalSiPMHitResponse_h

#include "SimCalorimetry/CaloSimAlgos/interface/CaloHitResponse.h"
#include "SimCalorimetry/HcalSimAlgos/interface/HcalSiPM.h"

#include <map>
#include <set>
#include <vector>

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

  virtual void initializeHits() override;

  virtual void finalizeHits(CLHEP::HepRandomEngine*) override;

  virtual void add(const PCaloHit& hit, CLHEP::HepRandomEngine*) override;

  virtual void add(const CaloSamples& signal);

  virtual void addPEnoise(CLHEP::HepRandomEngine* engine);

  virtual CaloSamples makeBlankSignal(const DetId & detId) const;

  virtual void setDetIds(const std::vector<DetId> & detIds);

protected:
  virtual CaloSamples makeSiPMSignal(DetId const& id, photonTimeHist const& photons, CLHEP::HepRandomEngine*);

private:
  HcalSiPM theSiPM;
  bool PreMixDigis;
  int nbins;
  double dt, invdt;

  photonTimeMap precisionTimedPhotons;

  const std::vector<DetId>* theDetIds;
};

#endif //HcalSimAlgos_HcalSiPMHitResponse_h
