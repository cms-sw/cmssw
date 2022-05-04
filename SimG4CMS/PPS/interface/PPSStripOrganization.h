#ifndef PPS_PPSStripOrganization_h
#define PPS_PPSStripOrganization_h

#include "globals.hh"
#include "SimG4CMS/PPS/interface/TotemRPVDetectorOrganization.h"

class PPSStripOrganization : public TotemRPVDetectorOrganization {
public:
  PPSStripOrganization(){};
  ~PPSStripOrganization() override = default;

  uint32_t unitID(const G4Step* aStep) override;
};

#endif  //PPS_PPSStripOrganization_h
