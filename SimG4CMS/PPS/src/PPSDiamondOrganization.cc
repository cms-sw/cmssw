// ////////////////////////////////////
// Author
// Seyed Mohsen Etesami  setesami@cern.ch
// //////////////////////////////////////////

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "SimG4CMS/PPS/interface/PPSDiamondOrganization.h"
#include "DataFormats/CTPPSDetId/interface/CTPPSDiamondDetId.h"
#include "G4VPhysicalVolume.hh"
#include "G4VTouchable.hh"
#include "G4Step.hh"

#include <iostream>

//******************************************************************** Constructor and destructor

PPSDiamondOrganization ::PPSDiamondOrganization()
    : theArm_(0), theStation_(0), theRoman_pot_(0), thePlane_(0), theDetector_(0) {}

uint32_t PPSDiamondOrganization::unitID(const G4Step* aStep) {
  G4VPhysicalVolume* physVol;
  const G4VTouchable* touch = aStep->GetPreStepPoint()->GetTouchable();
  bool foundVacuum = false;

  for (int ii = 0; ii < touch->GetHistoryDepth(); ii++) {
    physVol = touch->GetVolume(ii);

    if (physVol->GetName().contains("CTPPS_Diamond_Segment") || physVol->GetName().contains("CTPPS_UFSD_Segment")) {
      theDetector_ = physVol->GetCopyNo() % 100;
      thePlane_ = physVol->GetCopyNo() / 100;
      LogDebug("PPSSimDiamond") << "\n---------------------CTPPS_Diamond_Segment-------------------------------------"
                                   "------------------------------";
      LogDebug("PPSSimDiamond") << "\t\t\t\t\tDetector name " << physVol->GetName()
                                << " copynumber= " << physVol->GetCopyNo();
      LogDebug("PPSSimDiamond") << "\t\t\t\t\tdetector= " << theDetector_ << " plane= " << thePlane_ << " ii = " << ii;
    }

    else if (physVol->GetName().contains("Primary_Vacuum")) {
      int cpy_no = physVol->GetCopyNo();
      theArm_ = (cpy_no / 100) % 10;
      theStation_ = (cpy_no / 10) % 10;
      theRoman_pot_ = cpy_no % 10;
      foundVacuum = true;
    }
    LogDebug("PPSSimDiamond") << "Diamond physVol =" << physVol->GetName() << ", level=" << ii
                              << ", physVol->GetCopyNo()=" << physVol->GetCopyNo() << std::endl;
  }
  if (foundVacuum == false) {
    edm::LogError("PPSSimDiamond") << "Physical volume Primary_Vacuum not found. Cannot determine CTPPSDiamondDetId.";
  }
  return CTPPSDiamondDetId(theArm_, theStation_, theRoman_pot_, thePlane_, theDetector_).rawId();
}
