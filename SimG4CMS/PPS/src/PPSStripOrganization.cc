#include "SimG4CMS/PPS/interface/PPSStripOrganization.h"
#include "DataFormats/CTPPSDetId/interface/TotemRPDetId.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "G4VPhysicalVolume.hh"
#include "G4VTouchable.hh"
#include "G4Step.hh"

#include <iostream>

uint32_t PPSStripOrganization::unitID(const G4Step* aStep) {
  unsigned int arm = 0;
  unsigned int station = 0;
  unsigned int roman_pot = 0;

  const G4VTouchable* touch = aStep->GetPreStepPoint()->GetTouchable();
  G4VPhysicalVolume* physVol = touch->GetVolume(0);
  unsigned int detector = physVol->GetCopyNo();
  edm::LogVerbatim("PPSStripSim") << "PPSStripSim: PhysVol= " << physVol->GetName() << " coNum=" << detector;

  G4VPhysicalVolume* physVolVac = touch->GetVolume(2);
  if (nullptr == physVolVac) {
    edm::LogError("PPSStripSim") << "Physical volume RP_box_primary_vacuum not found for " << physVol->GetName()
                                 << "; cannot determine TotemRPDetId.";
  } else {
    int cpy_no = physVolVac->GetCopyNo();
    arm = (cpy_no / 100) % 10;
    station = (cpy_no / 10) % 10;
    roman_pot = cpy_no % 10;
  }
  edm::LogVerbatim("PPSStripSim") << "    arm=" << arm << " station=" << station << " roman_pot=" << roman_pot
                                  << " detector=" << detector;
  return TotemRPDetId(arm, station, roman_pot, detector).rawId();
}
