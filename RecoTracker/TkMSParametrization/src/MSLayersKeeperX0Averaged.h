#ifndef MSLayersKeeperX0Averaged_H
#define MSLayersKeeperX0Averaged_H

#include "MSLayersKeeper.h"
#include "FWCore/Framework/interface/EventSetup.h"
class dso_hidden MSLayersKeeperX0Averaged final : public MSLayersKeeper {
public:
  MSLayersKeeperX0Averaged() : isInitialised(false) { }
  ~MSLayersKeeperX0Averaged() override { }
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
