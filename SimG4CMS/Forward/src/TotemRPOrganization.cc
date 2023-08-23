// -*- C++ -*-
//
// Package:     Forward
// Class  :     TotemRPOrganization
//
// Implementation:
//     <Notes on implementation>
//
// Original Author:
//         Created:  Tue May 16 10:14:34 CEST 2006
//

// system include files

// user include files
#include "SimG4CMS/Forward/interface/TotemRPOrganization.h"
#include "SimG4CMS/Forward/interface/TotemNumberMerger.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "G4VPhysicalVolume.hh"
#include "G4VTouchable.hh"

//
// constructors and destructor
//
TotemRPOrganization ::TotemRPOrganization() { edm::LogVerbatim("ForwardSim") << "Creating TotemRPOrganization"; }

TotemRPOrganization ::~TotemRPOrganization() {}

//
// member functions
//

uint32_t TotemRPOrganization ::getUnitID(const G4Step* aStep) const {
  G4VPhysicalVolume* physVol;
  int32_t UNITA = 0;
  const G4VTouchable* touch = aStep->GetPreStepPoint()->GetTouchable();

#ifdef SCRIVI
  for (int ii = 0; ii < touch->GetHistoryDepth(); ii++) {
    physVol = touch->GetVolume(ii);
    LogDebug("ForwardSim") << "physVol=" << physVol->GetName() << ", level=" << ii
                           << ", physVol->GetCopyNo()=" << physVol->GetCopyNo();
  }
#endif

  physVol = touch->GetVolume(0);  //aStep->GetPreStepPoint()->GetPhysicalVolume();

  if (physVol->GetName() == "myRP")
    UNITA = (touch->GetVolume(5)->GetCopyNo()) * 1111;

#ifdef SCRIVI
  LogDebug("ForwardSim") << "\nUNITA-RP " << UNITA << "\n\n";
#endif
  return UNITA;
}
