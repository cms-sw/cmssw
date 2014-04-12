#ifndef MSLayersKeeperX0DetLayer_H
#define MSLayersKeeperX0DetLayer_H

#include "MSLayersKeeper.h"
#include "MultipleScatteringGeometry.h"
#include "FWCore/Framework/interface/EventSetup.h"
class dso_hidden MSLayersKeeperX0DetLayer GCC11_FINAL : public MSLayersKeeper {
public:
  MSLayersKeeperX0DetLayer() : isInitialised(false) { }
  virtual ~MSLayersKeeperX0DetLayer() { }
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
