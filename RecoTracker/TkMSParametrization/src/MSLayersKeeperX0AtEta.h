#ifndef MSLayersKeeperX0AtEta_H
#define MSLayersKeeperX0AtEta_H

#include "MSLayersKeeper.h"
#include "FWCore/Framework/interface/EventSetup.h"
class SumX0AtEtaDataProvider;
class MSLayersKeeperX0Averaged;

class dso_hidden MSLayersKeeperX0AtEta GCC11_FINAL : public MSLayersKeeper {
public:
  MSLayersKeeperX0AtEta() : isInitialised(false) { }
  ~MSLayersKeeperX0AtEta() { }
  void init(const edm::EventSetup &iSetup);
  const MSLayersAtAngle & layers(float cotTheta) const;

private:
  float eta(int idxBin) const;
  int idxBin(float eta) const;
  static void setX0(std::vector<MSLayer>&, float eta, const SumX0AtEtaDataProvider &);

private:
  bool isInitialised;
  int theHalfNBins; float theDeltaEta;
  std::vector<MSLayersAtAngle> theLayersData ;
  friend class MSLayersKeeperX0Averaged; 
};

#endif
