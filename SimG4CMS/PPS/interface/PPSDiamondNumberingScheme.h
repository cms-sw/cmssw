//  //////////////////////
// Author
// Seyed Mohsen Etesami setesami@cern.ch
// ////////////////////////

#ifndef PPS_PPSDiamondNumberingScheme_h
#define PPS_PPSDiamondNumberingScheme_h

#include "SimG4CMS/PPS/interface/PPSDiamondOrganization.h"

class PPSDiamondNumberingScheme : public PPSDiamondOrganization {
public:
  PPSDiamondNumberingScheme();
  ~PPSDiamondNumberingScheme() override;

  //  virtual unsigned int GetUnitID(const G4Step* aStep) const ;
};

#endif  //PPS_PPSDiamondNumberingScheme_h
