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
    : currentUnitID_(-1), currentArm_(-1), currentStation_(-1), currentRP_(-1), currentPlane_(-1) {
  edm::LogInfo("PPSSim") << "Creating PPSPixelOrganization";
}

//
// member functions
//

uint32_t PPSPixelOrganization ::unitID(const G4Step* aStep) {
  const G4VTouchable* touch = aStep->GetPreStepPoint()->GetTouchable();
  G4VPhysicalVolume* physVol;
  int ii = 0;

  for (ii = 0; ii < touch->GetHistoryDepth(); ii++) {
    physVol = touch->GetVolume(ii);

    edm::LogInfo("PPSSim") << "physVol=" << physVol->GetName() << ", level=" << ii
                           << ", physVol->GetCopyNo()=" << physVol->GetCopyNo();

    if (physVol->GetName().contains("Envelop")) {
      currentPlane_ = physVol->GetCopyNo() - 1;
    } else if (physVol->GetName() == "RP_box_primary_vacuum") {
      int cpy_no = physVol->GetCopyNo();
      currentArm_ = (cpy_no / 100) % 10;
      currentStation_ = (cpy_no / 10) % 10;
      currentRP_ = cpy_no % 10;
    }
  }

  edm::LogInfo("PPSSim") << currentArm_ << " " << currentRP_ << " " << currentPlane_;

  CTPPSPixelDetId id(currentArm_, currentStation_, currentRP_, currentPlane_);
  uint32_t kk = id.rawId();
  edm::LogInfo("PPSSim") << " ID " << kk;
  return id.rawId();
}
