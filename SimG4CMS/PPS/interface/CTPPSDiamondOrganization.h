// ///////////////////////
// Author 
// Seyed Mohsen Etesami setesami@cern.ch
// //////////////////////////////

#ifndef PPS_CTPPSDiamondOrganization_h
#define PPS_CTPPSDiamondOrganization_h 


#include "globals.hh"
#include "SimG4CMS/PPS/interface/PPSVDetectorOrganization.h"
#include "G4Step.hh"

class CTPPSDiamondOrganization : public PPSVDetectorOrganization
{
  public:
    CTPPSDiamondOrganization();
    virtual ~CTPPSDiamondOrganization();
 
    uint32_t GetUnitID(const G4Step* aStep);
    uint32_t GetUnitID(const G4Step* aStep) const;

  private:
    unsigned int theArm ;
    unsigned int theStation;
    unsigned int theRoman_pot;
    unsigned int thePlane;
    unsigned int theDetector ;

};



#endif //PPS_CTPPSDiamondOrganization_h
