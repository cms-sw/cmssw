#ifndef MSLayersKeeperX0Averaged_H
#define MSLayersKeeperX0Averaged_H

#include "FWCore/Utilities/interface/Visibility.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "RecoTracker/TkDetLayers/interface/GeometricSearchTracker.h"

#include "MSLayersKeeper.h"

class dso_hidden MSLayersKeeperX0Averaged final : public MSLayersKeeper {
public:
  MSLayersKeeperX0Averaged(const GeometricSearchTracker& tracker, const MagneticField& bfield);
  ~MSLayersKeeperX0Averaged() override = default;
  MSLayer layer(const DetLayer* layer) const override { return *theLayersData.findLayer(MSLayer(layer)); }
  const MSLayersAtAngle& layers(float cotTheta) const override { return theLayersData; }

private:
  MSLayersAtAngle theLayersData;
};
#endif
