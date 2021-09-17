#ifndef TotemRPVDetectorOrganization_h
#define TotemRPVDetectorOrganization_h

#include "G4Step.hh"

class TotemRPVDetectorOrganization {
public:
  TotemRPVDetectorOrganization(){};
  virtual ~TotemRPVDetectorOrganization(){};
  virtual uint32_t unitID(const G4Step* aStep) = 0;
};

#endif  //TotemRPVDetectorOrganization_h
