#ifndef MSLayersKeeperX0AtEta_H
#define MSLayersKeeperX0AtEta_H

#include "RecoTracker/TkMSParametrization/src/MSLayersKeeper.h"
class SumX0AtEtaDataProvider;
class MSLayersKeeperX0Averaged;

class MSLayersKeeperX0AtEta : public MSLayersKeeper {
public:
  MSLayersKeeperX0AtEta() : isInitialised(false) { }
  virtual ~MSLayersKeeperX0AtEta() { }
  virtual void init();
  virtual const MSLayersAtAngle & layers(float cotTheta) const;

private:
  float eta(int idxBin) const;
  int idxBin(float eta) const;
  void setX0(vector<MSLayer>&, float eta, const SumX0AtEtaDataProvider &) const;

private:
  bool isInitialised;
  int theHalfNBins; float theDeltaEta;
  vector<MSLayersAtAngle> theLayersData ;
  friend class MSLayersKeeperX0Averaged; 
};

#endif
