#ifndef MultipleScatteringGeometry_H
#define MultipleScatteringGeometry_H

#include <vector>
#include "RecoTracker/TkMSParametrization/interface/MSLayer.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "FWCore/Utilities/interface/GCC11Compatibility.h"

class DetLayer;

class dso_hidden MultipleScatteringGeometry {
public:
  MultipleScatteringGeometry(const edm::EventSetup &iSetup);
  std::vector<MSLayer> detLayers(float eta, 
			    float z,
			    const edm::EventSetup &iSetup ) const;
  std::vector<MSLayer> detLayers(const edm::EventSetup &iSetup) const;
  std::vector<MSLayer> otherLayers(float eta,const edm::EventSetup &iSetup) const;

 protected:
  std::vector<const DetLayer*> theLayers; 
  static const float beamPipeR, endflangesZ, supportR;

};
#endif
