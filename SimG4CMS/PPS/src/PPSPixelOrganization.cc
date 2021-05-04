// -*- C++ -*-
//
// Package:     PPS
// Class  :     PPSPixelOrganization
//
// Implementation:
//     <Notes on implementation>
//

// user include files
#include "SimG4CMS/PPS/interface/PPSPixelOrganization.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/CTPPSDetId/interface/CTPPSPixelDetId.h"
#include "G4VPhysicalVolume.hh"
#include "G4VTouchable.hh"

//
// constructors and destructor
//
PPSPixelOrganization ::PPSPixelOrganization()
    : currentUnitID_(0), currentArm_(0), currentStation_(0), currentRP_(0), currentPlane_(0) {
  edm::LogInfo("PPSSim") << "Creating PPSPixelOrganization";
}

//
// member functions
//

uint32_t PPSPixelOrganization ::unitID(const G4Step* aStep) {
  const G4VTouchable* touch = aStep->GetPreStepPoint()->GetTouchable();
  G4VPhysicalVolume* physVol;
  int ii = 0;
  bool foundEnvelop = false;
  bool foundPhysVol = false;

  while (ii < touch->GetHistoryDepth() && (foundEnvelop == false || foundPhysVol == false)) {
    physVol = touch->GetVolume(ii);

    edm::LogInfo("PPSSim") << "physVol=" << physVol->GetName() << ", level=" << ii
                           << ", physVol->GetCopyNo()=" << physVol->GetCopyNo();

    if (physVol->GetName().contains("Envelop")) {
      currentPlane_ = physVol->GetCopyNo() - 1;
      foundEnvelop = true;
    } else if (physVol->GetName().contains("RP_box_primary_vacuum")) {
      int cpy_no = physVol->GetCopyNo();
      currentArm_ = (cpy_no / 100) % 10;
      currentStation_ = (cpy_no / 10) % 10;
      currentRP_ = cpy_no % 10;
      foundPhysVol = true;
    }
    ++ii;
  }

  if (foundPhysVol) {
    edm::LogInfo("PPSSim") << "Arm, RP, plane = " << currentArm_ << " " << currentRP_ << " " << currentPlane_;
  } else {
    edm::LogError("PPSSim") << "Physical volume RP_box_primary_vacuum not found. Cannot determine CTPPSPixelDetId.";
  }

  CTPPSPixelDetId id(currentArm_, currentStation_, currentRP_, currentPlane_);
  uint32_t kk = id.rawId();
  edm::LogInfo("PPSSim") << " ID " << kk;
  return id.rawId();
}
