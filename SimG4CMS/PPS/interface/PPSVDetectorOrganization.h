///////////////////////////////////////////////////////////////////////////////
// File: PPSVDetectorOrganization.h
// Description: Base class for numbering scheme of PPS
///////////////////////////////////////////////////////////////////////////////
#ifndef _PPS_VDetectorOrganization_h
#define _PPS_VDetectorOrganization_h

#include "G4Step.hh"

class PPSVDetectorOrganization {
public:
  PPSVDetectorOrganization(){};
  virtual ~PPSVDetectorOrganization(){};
  virtual uint32_t unitID(const G4Step* aStep) = 0;
};

#endif
