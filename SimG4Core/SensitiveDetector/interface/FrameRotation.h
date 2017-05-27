#ifndef SimG4Core_FrameRotation_H
#define SimG4Core_FrameRotation_H

#include "DataFormats/GeometryVector/interface/LocalPoint.h"
#include "DataFormats/GeometryVector/interface/LocalVector.h"

#include "G4VPhysicalVolume.hh"

class FrameRotation 
{
public:
    virtual ~FrameRotation() = default;
    virtual Local3DPoint transformPoint(Local3DPoint &,G4VPhysicalVolume *) const = 0;
};

#endif
