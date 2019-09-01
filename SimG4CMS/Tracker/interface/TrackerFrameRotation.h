#ifndef SimG4CMS_TrackerFrameRotation_H
#define SimG4CMS_TrackerFrameRotation_H

#include "SimG4Core/SensitiveDetector/interface/FrameRotation.h"

/**
 * To be used for test beam etc. Note: if the sensitive detecor is created without an organization, force this one
 */

class G4VPhysicalVolume;

class TrackerFrameRotation : public FrameRotation {
public:
  ~TrackerFrameRotation() override = default;
  Local3DPoint transformPoint(const Local3DPoint &, const G4VPhysicalVolume *v = nullptr) const final;
};

#endif
