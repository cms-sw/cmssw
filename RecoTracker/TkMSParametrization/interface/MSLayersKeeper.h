#ifndef MSLayersKeeper_H
#define MSLayersKeeper_H

class DetLayer;

#include "RecoTracker/TkMSParametrization/interface/MSLayer.h"
#include "RecoTracker/TkMSParametrization/interface/MSLayersAtAngle.h"
#include "FWCore/Framework/interface/EventSetup.h"

class MSLayersKeeper {
public:
  virtual ~MSLayersKeeper() { }
  virtual MSLayer layer(const DetLayer* dl) const 
    { return MSLayer(dl,DataX0(this)); }
  virtual const MSLayersAtAngle & layers(float cotTheta) const = 0;  
  virtual void init(const edm::EventSetup &iSetup) { }
protected:
  typedef MSLayer::DataX0 DataX0;
  const DataX0 & getDataX0(const MSLayer & l) const { return l.theX0Data; }
  void  setDataX0(MSLayer & l, DataX0 x0Data) const { l.theX0Data = x0Data; }  
};

#endif
