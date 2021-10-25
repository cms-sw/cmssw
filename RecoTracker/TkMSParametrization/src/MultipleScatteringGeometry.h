#ifndef MultipleScatteringGeometry_H
#define MultipleScatteringGeometry_H

#include <vector>
#include "FWCore/Utilities/interface/Visibility.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "RecoTracker/TkDetLayers/interface/GeometricSearchTracker.h"
#include "RecoTracker/TkMSParametrization/interface/MSLayer.h"

class DetLayer;

class dso_hidden MultipleScatteringGeometry {
public:
  MultipleScatteringGeometry(const GeometricSearchTracker &tracker);
  std::vector<MSLayer> detLayers(float eta, float z, const MagneticField &bfield) const;
  std::vector<MSLayer> detLayers() const;
  std::vector<MSLayer> otherLayers(float eta) const;

protected:
  std::vector<const DetLayer *> theLayers;
  static const float beamPipeR, endflangesZ, supportR;
};
#endif
