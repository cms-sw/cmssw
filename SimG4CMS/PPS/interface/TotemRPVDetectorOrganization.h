#ifndef TotemRPVDetectorOrganization_h
#define TotemRPVDetectorOrganization_h

#include "G4Step.hh"
#include <boost/cstdint.hpp>

class TotemRPVDetectorOrganization {

public:
  TotemRPVDetectorOrganization(){};
  virtual ~TotemRPVDetectorOrganization(){};   
  virtual uint32_t GetUnitID(const  G4Step* aStep) const =0;
};      


#endif  //TotemRPVDetectorOrganization_h
