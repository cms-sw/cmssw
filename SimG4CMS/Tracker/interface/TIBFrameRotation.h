#ifndef SimG4CMS_TIBFrameRotation_H
#define SimG4CMS_TIBFrameRotation_H

#include "SimG4Core/SensitiveDetector/interface/FrameRotation.h"

#include "G4StepPoint.hh"
#include "G4VPhysicalVolume.hh"

class TIBFrameRotation : public FrameRotation 
{
public:
    virtual Local3DPoint transformPoint(Local3DPoint &,G4VPhysicalVolume *) const;
};

#endif
