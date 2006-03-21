#ifndef MultipleScatteringGeometry_H
#define MultipleScatteringGeometry_H

#include <vector>
#include "RecoTracker/TkMSParametrization/interface/MSLayer.h"

class DetLayer;

class MultipleScatteringGeometry {
public:
  MultipleScatteringGeometry();
  vector<MSLayer> detLayers(float eta, float z = 0.) const;
  vector<MSLayer> detLayers() const;
  vector<MSLayer> otherLayers(float eta) const;
protected:
  vector<const DetLayer*> theLayers; 
  static const float beamPipeR, endflangesZ, supportR;
};
#endif
