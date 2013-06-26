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
// $Id: TotemRPOrganization.cc,v 1.1 2006/05/17 16:18:58 sunanda Exp $
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
TotemRPOrganization :: TotemRPOrganization() :
  _needUpdateUnitID(false), _needUpdateData(false), _currentUnitID(-1),
  _currentPlane(-1), _currentCSC(-1), _currentLayer(-1) {

  edm::LogInfo("ForwardSim") << "Creating TotemRPOrganization";
}

TotemRPOrganization :: ~TotemRPOrganization() {
}

//
// member functions
//

uint32_t TotemRPOrganization :: GetUnitID(const G4Step* aStep) const {
  return const_cast<TotemRPOrganization *>(this)->GetUnitID(aStep);
}

uint32_t TotemRPOrganization :: GetUnitID(const G4Step* aStep) {

  G4VPhysicalVolume* physVol;
  int32_t UNITA=0;
  const G4VTouchable* touch = aStep->GetPreStepPoint()->GetTouchable();
  int ii =0;
  for( ii = 0; ii < touch->GetHistoryDepth(); ii++ ){
    physVol = touch->GetVolume(ii);
   
#ifdef SCRIVI
    LogDebug("ForwardSim") << "physVol=" << physVol->GetName() << ", level="
			   << ii  << ", physVol->GetCopyNo()=" 
			   << physVol->GetCopyNo();
#endif
    if (physVol->GetName() == "myRP") _currentDetectorPosition = 3;
      
  }
  physVol= touch->GetVolume(0);//aStep->GetPreStepPoint()->GetPhysicalVolume();
  
  if(physVol->GetName() == "myRP") UNITA=(touch->GetVolume(5)->GetCopyNo())*1111;

#ifdef SCRIVI
  LogDebug("ForwardSim") << "\nUNITA-RP " << UNITA << "\n\n";
#endif
  return UNITA;
}
