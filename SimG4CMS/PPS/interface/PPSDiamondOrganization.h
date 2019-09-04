// Author
// Seyed Mohsen Etesami setesami@cern.ch

#ifndef PPS_PPSDiamondOrganization_h
#define PPS_PPSDiamondOrganization_h

#include "globals.hh"
#include "SimG4CMS/PPS/interface/PPSVDetectorOrganization.h"
#include "G4Step.hh"

class PPSDiamondOrganization : public PPSVDetectorOrganization {
public:
  PPSDiamondOrganization();
  ~PPSDiamondOrganization() override = default;

  uint32_t unitID(const G4Step* aStep) override;

private:
  unsigned int theArm_;
  unsigned int theStation_;
  unsigned int theRoman_pot_;
  unsigned int thePlane_;
  unsigned int theDetector_;
};

#endif  //PPS_PPSDiamondOrganization_h
