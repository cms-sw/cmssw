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
  edm::LogVerbatim("PPSSim") << "Creating PPSPixelOrganization";
}

//
// member functions
//

uint32_t PPSPixelOrganization ::unitID(const G4Step* aStep) {
  const G4VTouchable* touch = aStep->GetPreStepPoint()->GetTouchable();
  G4VPhysicalVolume* physVol = touch->GetVolume(0);
  int coNum = physVol->GetCopyNo();
  edm::LogVerbatim("PPSPixelSim") << "PPSPixelSim: PhysVol= " << physVol->GetName() << " coNum=" << coNum;
  currentPlane_ = coNum - 1;

  G4VPhysicalVolume* physVolVac = touch->GetVolume(3);
  if (nullptr == physVolVac) {
    edm::LogError("PPSPixelSim") << "Physical volume RP_box_primary_vacuum not found for " << physVol->GetName()
                                 << "; cannot determine CTPPSPixelDetId.";
  } else {
    int cpy_no = physVolVac->GetCopyNo();
    currentArm_ = (cpy_no / 100) % 10;
    currentStation_ = (cpy_no / 10) % 10;
    currentRP_ = cpy_no % 10;
  }

  edm::LogVerbatim("PPSPixelSim") << "    arm=" << currentArm_ << " station=" << currentStation_
                                  << " roman_pot=" << currentRP_ << " detector=" << currentPlane_;
  CTPPSPixelDetId id(currentArm_, currentStation_, currentRP_, currentPlane_);
  uint32_t kk = id.rawId();
  edm::LogVerbatim("PPSPixelSim") << "PPSPixelSim: ID=" << kk;
  return kk;
}
