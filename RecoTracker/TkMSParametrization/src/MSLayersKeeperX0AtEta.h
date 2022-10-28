#ifndef MSLayersKeeperX0AtEta_H
#define MSLayersKeeperX0AtEta_H

#include "FWCore/Utilities/interface/Visibility.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "RecoTracker/TkDetLayers/interface/GeometricSearchTracker.h"

#include "MSLayersKeeper.h"

class SumX0AtEtaDataProvider;
class MSLayersKeeperX0Averaged;

class dso_hidden MSLayersKeeperX0AtEta final : public MSLayersKeeper {
public:
  MSLayersKeeperX0AtEta(const GeometricSearchTracker &tracker, const MagneticField &bfield);
  ~MSLayersKeeperX0AtEta() override;
  const MSLayersAtAngle &layers(float cotTheta) const override;

private:
  float eta(int idxBin) const;
  int idxBin(float eta) const;
  static void setX0(std::vector<MSLayer> &, float eta, const SumX0AtEtaDataProvider &);

private:
  int theHalfNBins;
  float theDeltaEta;
  std::vector<MSLayersAtAngle> theLayersData;
  friend class MSLayersKeeperX0Averaged;
};

#endif
