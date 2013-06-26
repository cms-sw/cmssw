#ifndef MSLayersKeeperX0Averaged_H
#define MSLayersKeeperX0Averaged_H

#include "MSLayersKeeper.h"
#include "FWCore/Framework/interface/EventSetup.h"
class dso_hidden MSLayersKeeperX0Averaged GCC11_FINAL : public MSLayersKeeper {
public:
  MSLayersKeeperX0Averaged() : isInitialised(false) { }
  virtual ~MSLayersKeeperX0Averaged() { }
  virtual void init(const edm::EventSetup &iSetup);
  virtual MSLayer layer(const DetLayer* layer) const
    {return *theLayersData.findLayer(MSLayer(layer)); }
  virtual const MSLayersAtAngle & layers(float cotTheta) const 
    {return theLayersData;}

private:
  bool isInitialised;
  MSLayersAtAngle theLayersData;
};
#endif
