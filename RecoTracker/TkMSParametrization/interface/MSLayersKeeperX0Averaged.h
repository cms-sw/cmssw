#ifndef MSLayersKeeperX0Averaged_H
#define MSLayersKeeperX0Averaged_H

#include "RecoTracker/TkMSParametrization/interface/MSLayersKeeper.h"
#include "FWCore/Framework/interface/EventSetup.h"
class MSLayersKeeperX0Averaged : public MSLayersKeeper {
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
