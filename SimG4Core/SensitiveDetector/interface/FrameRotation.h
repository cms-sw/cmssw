#ifndef SimG4Core_SensitiveDetector_FrameRotation_H
#define SimG4Core_SensitiveDetector_FrameRotation_H

#include "DataFormats/GeometryVector/interface/LocalPoint.h"
//#include "G4VPhysicalVolume.hh"

class G4VPhysicalVolume;

class FrameRotation 
{
public:
    virtual ~FrameRotation() = default;
    virtual Local3DPoint transformPoint(const Local3DPoint &,const G4VPhysicalVolume *v=nullptr) const = 0;
};

#endif
