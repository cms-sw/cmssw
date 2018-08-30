//  //////////////////////
// Author 
// Seyed Mohsen Etesami setesami@cern.ch
// ////////////////////////

#ifndef PPS_CTPPSDiamondNumberingScheme_h
#define PPS_CTPPSDiamondNumberingScheme_h

#include "SimG4CMS/PPS/interface/CTPPSDiamondOrganization.h"

class CTPPSDiamondNumberingScheme : public CTPPSDiamondOrganization 
{
  public:
    CTPPSDiamondNumberingScheme();
    ~CTPPSDiamondNumberingScheme() override;
	 
    //  virtual unsigned int GetUnitID(const G4Step* aStep) const ;

};

#endif //PPS_CTPPSDiamondNumberingScheme_h
