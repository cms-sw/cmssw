#ifndef MSLayersKeeperX0DetLayer_H
#define MSLayersKeeperX0DetLayer_H

#include "RecoTracker/TkMSParametrization/interface/MSLayersKeeper.h"
#include "RecoTracker/TkMSParametrization/interface/MultipleScatteringGeometry.h"
#include "FWCore/Framework/interface/EventSetup.h"
class MSLayersKeeperX0DetLayer : public MSLayersKeeper {
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
/*   class MSLayersKeeperX0DetLayerGeom : public MultipleScatteringGeometry {   */
  
/*   public:  */
/*     vector<MSLayer> detLayers() const; */
/*   }; */
};
#endif
