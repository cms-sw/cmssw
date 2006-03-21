#ifndef MultipleScatteringGeometry_H
#define MultipleScatteringGeometry_H

#include <vector>
#include "RecoTracker/TkMSParametrization/interface/MSLayer.h"
#include "FWCore/Framework/interface/EventSetup.h"
class DetLayer;

class MultipleScatteringGeometry {
public:
  MultipleScatteringGeometry(const edm::EventSetup &iSetup);
  vector<MSLayer> detLayers(float eta, 
			    float z,
			    const edm::EventSetup &iSetup ) const;
  vector<MSLayer> detLayers(const edm::EventSetup &iSetup) const;
  vector<MSLayer> otherLayers(float eta,const edm::EventSetup &iSetup) const;

 protected:
  vector<const DetLayer*> theLayers; 
  static const float beamPipeR, endflangesZ, supportR;

};
#endif
