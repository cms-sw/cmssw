#ifndef SimG4Core_SensitiveDetector_FrameRotation_H
#define SimG4Core_SensitiveDetector_FrameRotation_H

#include "DataFormats/GeometryVector/interface/LocalPoint.h"

class G4VPhysicalVolume;

class FrameRotation 
{
public:

  // from Geant4 unit of coordinates to CMS
  static constexpr double invcm = 0.1;

  virtual ~FrameRotation() = default;

  virtual Local3DPoint transformPoint(const Local3DPoint &,const G4VPhysicalVolume *v=nullptr) const = 0;
};

#endif
