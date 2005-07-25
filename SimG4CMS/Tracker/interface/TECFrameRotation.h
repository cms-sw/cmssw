#ifndef SimG4CMS_TECFrameRotation_H
#define SimG4CMS_TECFrameRotation_H

#include "SimG4Core/SensitiveDetector/interface/FrameRotation.h"

#include "G4StepPoint.hh"
#include "G4VPhysicalVolume.hh"

class TECFrameRotation : public FrameRotation 
{
public:
    virtual Local3DPoint transformPoint(Local3DPoint &,G4VPhysicalVolume *) const;
};

#endif
