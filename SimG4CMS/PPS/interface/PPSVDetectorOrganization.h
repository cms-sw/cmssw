///////////////////////////////////////////////////////////////////////////////
// File: PPSVDetectorOrganization.h
// Description: Base class for numbering scheme of PPS
///////////////////////////////////////////////////////////////////////////////
#ifndef _PPS_VDetectorOrganization_h
#define _PPS_VDetectorOrganization_h

#include "G4Step.hh"
#include <boost/cstdint.hpp>

class PPSVDetectorOrganization {

public:
  PPSVDetectorOrganization(){};
  virtual ~PPSVDetectorOrganization(){};   
  virtual uint32_t GetUnitID(const  G4Step* aStep) const =0;
};      


#endif


