// -*- C++ -*-
#ifndef HcalSimAlgos_HcalSiPMHitResponse_h
#define HcalSimAlgos_HcalSiPMHitResponse_h

#include "SimCalorimetry/CaloSimAlgos/interface/CaloHitResponse.h"
#include "SimCalorimetry/HcalSimAlgos/interface/HcalSiPM.h"
#include "SimCalorimetry/HcalSimAlgos/interface/HcalSiPMShape.h"

#include <map>
#include <set>
#include <vector>

namespace CLHEP {
  class HepRandomEngine;
}

class PCaloHitCompareTimes {
public:
  bool operator()(const PCaloHit* a, const PCaloHit* b) const { return a->time() < b->time(); }
};

class HcalSiPMHitResponse final : public CaloHitResponse {
public:
  HcalSiPMHitResponse(const CaloVSimParameterMap* parameterMap,
                      const CaloShapes* shapes,
                      bool PreMix1 = false,
                      bool HighFidelity = true);

  ~HcalSiPMHitResponse() override;

  typedef std::vector<unsigned int> photonTimeHist;
  typedef std::map<DetId, photonTimeHist> photonTimeMap;

  void initializeHits() override;

  void finalizeHits(CLHEP::HepRandomEngine*) override;

  void add(const PCaloHit& hit, CLHEP::HepRandomEngine*) override;

  void add(const CaloSamples& signal) override;

  virtual void addPEnoise(CLHEP::HepRandomEngine* engine);

  virtual CaloSamples makeBlankSignal(const DetId& detId) const;

  virtual void setDetIds(const std::vector<DetId>& detIds);

  virtual int getReadoutFrameSize(const DetId& id) const;

protected:
  virtual CaloSamples makeSiPMSignal(DetId const& id, photonTimeHist const& photons, CLHEP::HepRandomEngine*);

private:
  HcalSiPM theSiPM;
  bool PreMixDigis;
  bool HighFidelityPreMix;
  int nbins;
  double dt, invdt;

  photonTimeMap precisionTimedPhotons;

  const std::vector<DetId>* theDetIds;

  std::map<int, HcalSiPMShape> shapeMap;
};

#endif  //HcalSimAlgos_HcalSiPMHitResponse_h
