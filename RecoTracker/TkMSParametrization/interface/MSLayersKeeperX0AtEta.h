#ifndef MSLayersKeeperX0AtEta_H
#define MSLayersKeeperX0AtEta_H

#include "RecoTracker/TkMSParametrization/interface/MSLayersKeeper.h"
#include "FWCore/Framework/interface/EventSetup.h"
class SumX0AtEtaDataProvider;
class MSLayersKeeperX0Averaged;

class MSLayersKeeperX0AtEta : public MSLayersKeeper {
public:
  MSLayersKeeperX0AtEta() : isInitialised(false) { }
  virtual ~MSLayersKeeperX0AtEta() { }
  virtual void init(const edm::EventSetup &iSetup);
  virtual const MSLayersAtAngle & layers(float cotTheta) const;

private:
  float eta(int idxBin) const;
  int idxBin(float eta) const;
  void setX0(std::vector<MSLayer>&, float eta, const SumX0AtEtaDataProvider &) const;

private:
  bool isInitialised;
  int theHalfNBins; float theDeltaEta;
  std::vector<MSLayersAtAngle> theLayersData ;
  friend class MSLayersKeeperX0Averaged; 
};

#endif
