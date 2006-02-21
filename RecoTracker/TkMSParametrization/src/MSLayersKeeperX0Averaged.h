#ifndef MSLayersKeeperX0Averaged_H
#define MSLayersKeeperX0Averaged_H

#include "RecoTracker/TkMSParametrization/src/MSLayersKeeper.h"

class MSLayersKeeperX0Averaged : public MSLayersKeeper {
public:
  MSLayersKeeperX0Averaged() : isInitialised(false) { }
  virtual ~MSLayersKeeperX0Averaged() { }
  virtual void init();
  virtual MSLayer layer(const DetLayer* layer) const
    {return *theLayersData.findLayer(MSLayer(layer)); }
  virtual const MSLayersAtAngle & layers(float cotTheta) const 
    {return theLayersData;}

private:
  bool isInitialised;
  MSLayersAtAngle theLayersData;
};
#endif
