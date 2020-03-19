#ifndef PPS_PPSStripOrganization_h
#define PPS_PPSStripOrganization_h

#include "globals.hh"
#include "SimG4CMS/PPS/interface/TotemRPVDetectorOrganization.h"
#include "G4Step.hh"

class PPSStripOrganization : public TotemRPVDetectorOrganization {
public:
  inline PPSStripOrganization();
  ~PPSStripOrganization() override = default;

  uint32_t unitID(const G4Step* aStep) override;
};

inline PPSStripOrganization ::PPSStripOrganization() = default;

#endif  //PPS_PPSStripOrganization_h
