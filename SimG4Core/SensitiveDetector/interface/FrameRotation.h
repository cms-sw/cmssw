#ifndef SimG4Core_FrameRotation_H
#define SimG4Core_FrameRotation_H

#include "Geometry/Vector/interface/LocalPoint.h"
#include "Geometry/Vector/interface/LocalVector.h"

#include "G4VPhysicalVolume.hh"

class FrameRotation 
{
public:
    virtual Local3DPoint transformPoint(Local3DPoint &,G4VPhysicalVolume *) const = 0;
};

#endif
