#ifndef MSLayersKeeperX0DetLayer_H
#define MSLayersKeeperX0DetLayer_H

#include "MSLayersKeeper.h"
#include "MultipleScatteringGeometry.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Utilities/interface/Visibility.h"

class dso_hidden MSLayersKeeperX0DetLayer final : public MSLayersKeeper {
public:
  MSLayersKeeperX0DetLayer();
  ~MSLayersKeeperX0DetLayer() override = default;
  MSLayer layer(const DetLayer* layer) const override { return *theLayersData.findLayer(MSLayer(layer)); }
  const MSLayersAtAngle& layers(float cotTheta) const override { return theLayersData; }

private:
  MSLayersAtAngle theLayersData;
};
#endif
