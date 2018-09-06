// ////////////////////////////////////
// Author
// Seyed Mohsen Etesami  setesami@cern.ch
// //////////////////////////////////////////

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "SimG4CMS/PPS/interface/CTPPSDiamondOrganization.h"
#include "DataFormats/CTPPSDetId/interface/CTPPSDiamondDetId.h"
#include "G4VPhysicalVolume.hh"
#include "G4VTouchable.hh"
#include "G4Step.hh"

#include <iostream>

//******************************************************************** Constructor and destructor

CTPPSDiamondOrganization :: CTPPSDiamondOrganization():
theArm(-1),theStation(-1),theRoman_pot(-1),thePlane(-1),theDetector(-1)
{
}

CTPPSDiamondOrganization :: ~CTPPSDiamondOrganization()
{
}


uint32_t CTPPSDiamondOrganization::GetUnitID(const G4Step* aStep) const
{
  return const_cast<CTPPSDiamondOrganization *>(this)->GetUnitID(aStep);
}



uint32_t CTPPSDiamondOrganization::GetUnitID(const G4Step* aStep)
{
  G4VPhysicalVolume* physVol;             
  const G4VTouchable* touch = aStep->GetPreStepPoint()->GetTouchable();

  for(int ii = 0; ii < touch->GetHistoryDepth(); ii++ )
  {
    physVol = touch->GetVolume(ii);

    if(physVol->GetName() == "CTPPS_Diamond_Segment" || physVol->GetName() == "CTPPS_UFSD_Segment")
    {
      theDetector = physVol->GetCopyNo()%100;
      thePlane= physVol->GetCopyNo()/100 ;
      LogDebug("CTPPSSimDiamond")<< "\n---------------------CTPPS_Diamond_Segment-------------------------------------------------------------------";
      LogDebug("CTPPSSimDiamond")<<"\t\t\t\t\tDetector name "<<physVol->GetName()<<" copynumber= "<<physVol->GetCopyNo();
      LogDebug("CTPPSSimDiamond")<<"\t\t\t\t\tdetector= "<<theDetector<<" plane= "<<thePlane<< " ii = " << ii ;
    }

    else if(physVol->GetName() == "Primary_Vacuum" )   
    {                        
      int cpy_no = physVol->GetCopyNo();
      theArm = (cpy_no/100)%10;
      theStation = (cpy_no/10)%10;
      theRoman_pot =cpy_no%10 ;
    }
    LogDebug("CTPPSSimDiamond")<<"Diamond" << "physVol =" << physVol->GetName()<< ", level=" << ii
			      << ", physVol->GetCopyNo()=" << physVol->GetCopyNo()<<std::endl;
  } 
  return CTPPSDiamondDetId(theArm, theStation, theRoman_pot, thePlane, theDetector).rawId(); 
}
