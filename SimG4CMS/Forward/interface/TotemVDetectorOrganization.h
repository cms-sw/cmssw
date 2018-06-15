///////////////////////////////////////////////////////////////////////////////
// File: TotemVDetectorOrganization.h
// Description: Base class for numbering scheme of Totem
///////////////////////////////////////////////////////////////////////////////
#ifndef Forward_TotemVDetectorOrganization_h
#define Forward_TotemVDetectorOrganization_h

#include "G4Step.hh"
#include <cstdint>

class TotemVDetectorOrganization {

public:
  TotemVDetectorOrganization(){};
  virtual ~TotemVDetectorOrganization(){};   
  virtual uint32_t GetUnitID(const  G4Step* aStep) const =0;
};      


#endif
