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
  const G4VTouchable* touch = aStep->GetPreStepPoint()->GetTouchable();
  G4VPhysicalVolume* physVol = touch->GetVolume(0);
  int coNum = physVol->GetCopyNo();
  edm::LogVerbatim("PPSSimDiamond") << " %%%% PhysVol: " << physVol->GetName() << " coNum=" << coNum;
  theDetector_ = coNum % 100;
  thePlane_ = coNum / 100;

  G4VPhysicalVolume* physVolVac = touch->GetVolume(3);
  if (nullptr == physVolVac) {
    edm::LogError("PPSSimDiamond") << "Physical volume Primary_Vacuum not found for " << physVol->GetName()
                                   << "; cannot determine CTPPSDiamondDetId.";
  } else {
    int cpy_no = physVolVac->GetCopyNo();
    theArm_ = (cpy_no / 100) % 10;
    theStation_ = (cpy_no / 10) % 10;
    theRoman_pot_ = cpy_no % 10;
  }
  return CTPPSDiamondDetId(theArm_, theStation_, theRoman_pot_, thePlane_, theDetector_).rawId();
}
