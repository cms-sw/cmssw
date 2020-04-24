#ifndef MSLayersKeeperX0DetLayer_H
#define MSLayersKeeperX0DetLayer_H

#include "MSLayersKeeper.h"
#include "MultipleScatteringGeometry.h"
#include "FWCore/Framework/interface/EventSetup.h"
class dso_hidden MSLayersKeeperX0DetLayer final : public MSLayersKeeper {
public:
  MSLayersKeeperX0DetLayer() : isInitialised(false) { }
  ~MSLayersKeeperX0DetLayer() override { }
  void init(const edm::EventSetup &iSetup) override;
  MSLayer layer(const DetLayer* layer) const override
    {return *theLayersData.findLayer(MSLayer(layer)); }
  const MSLayersAtAngle & layers(float cotTheta) const override
    {return theLayersData;}

private:
  bool isInitialised;
  MSLayersAtAngle theLayersData;
};
#endif
