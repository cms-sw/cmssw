// -*- C++ -*-
//
// Package:     Forward
// Class  :     TotemT2OrganizationGem
//
// Implementation:
//     <Notes on implementation>
//
// Original Author: 
//         Created:  Tue May 16 10:14:34 CEST 2006
// $Id: TotemT2OrganizationGem.cc,v 1.1 2006/05/17 16:18:58 sunanda Exp $
//

// system include files

// user include files
#include "SimG4CMS/Forward/interface/TotemT2OrganizationGem.h"
#include "SimG4CMS/Forward/interface/TotemNumberMerger.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "G4VPhysicalVolume.hh"
#include "G4VTouchable.hh" 

//
// constructors and destructor
//

TotemT2OrganizationGem :: TotemT2OrganizationGem() :
  _needUpdateUnitID(false), _needUpdateData(false),
  _currentUnitID(-1), _currentPlane(-1), _currentCSC(-1),
  _currentLayer(-1) {
  edm::LogInfo("ForwardSim") << "Creating TotemT2OrganizationGem";
}

TotemT2OrganizationGem :: ~TotemT2OrganizationGem() {
}

uint32_t TotemT2OrganizationGem :: GetUnitID(const G4Step* aStep) const {
  return const_cast<TotemT2OrganizationGem *>(this)->GetUnitID(aStep);
}

uint32_t TotemT2OrganizationGem :: GetUnitID(const G4Step* aStep) {

 G4VPhysicalVolume* physVol;
 uint32_t UNITA = 0;
 const G4VTouchable* touch = aStep->GetPreStepPoint()->GetTouchable();
 int ii =0;
 for( ii = 0; ii < touch->GetHistoryDepth(); ii++ ){
   physVol = touch->GetVolume(ii);

#ifdef SCRIVI
   LogDebug("ForwardSim") << "physVol=" << physVol->GetName() << ", level=" 
			  << ii << ", physVol->GetCopyNo()=" 
			  << physVol->GetCopyNo();
#endif
   if (physVol->GetName() == "TotemT2gem") _currentDetectorPosition = 3;
 }
  
 physVol= touch->GetVolume(0);
  
 if(physVol->GetName() == "TotemT2gem")UNITA = 10 + physVol->GetCopyNo()  ;
 if(physVol->GetName() == "TotemT2gem_supporto")UNITA =20 +  physVol->GetCopyNo();
 if(physVol->GetName() == "TotemT2gem_detector7r")UNITA = 100 + physVol->GetCopyNo() + (touch->GetVolume(2)->GetCopyNo())*1000;
 if(physVol->GetName() == "TotemT2gem_HC7r")UNITA = 200 + touch->GetVolume(1)->GetCopyNo()+ (touch->GetVolume(3)->GetCopyNo())*1000;
 if(physVol->GetName() == "TotemT2gem_drift7r")UNITA = 300 + touch->GetVolume(1)->GetCopyNo()+ (touch->GetVolume(3)->GetCopyNo())*1000;
 if(physVol->GetName() == "TotemT2gem_driftspace7r")UNITA = 400 + touch->GetVolume(1)->GetCopyNo()+ (touch->GetVolume(3)->GetCopyNo())*1000;
 if(physVol->GetName() == "TotemT2gem_GEMa7r")UNITA = 500 + touch->GetVolume(1)->GetCopyNo()+ (touch->GetVolume(3)->GetCopyNo())*1000;
 if(physVol->GetName() == "TotemT2gem_GEMb7r")UNITA = 600 + touch->GetVolume(1)->GetCopyNo()+ (touch->GetVolume(3)->GetCopyNo())*1000;
 if(physVol->GetName() == "TotemT2gem_GEMc7r")UNITA = 700 + touch->GetVolume(1)->GetCopyNo()+ (touch->GetVolume(3)->GetCopyNo())*1000;
 if(physVol->GetName() == "TotemT2gem_GAS7r")UNITA = 800 + touch->GetVolume(1)->GetCopyNo()+ (touch->GetVolume(3)->GetCopyNo())*1000;
 if(physVol->GetName() == "TotemT2gem_GEMa17r")UNITA = 900 + touch->GetVolume(1)->GetCopyNo()+ (touch->GetVolume(3)->GetCopyNo())*1000;
 if(physVol->GetName() == "TotemT2gem_GEMb17r")UNITA = 1000 + touch->GetVolume(1)->GetCopyNo()+ (touch->GetVolume(3)->GetCopyNo())*1000;
 if(physVol->GetName() == "TotemT2gem_GEMc17r")UNITA = 1100 + touch->GetVolume(1)->GetCopyNo()+ (touch->GetVolume(3)->GetCopyNo())*1000;
 if(physVol->GetName() == "TotemT2gem_GAS17r")UNITA = 1200 + touch->GetVolume(1)->GetCopyNo()+ (touch->GetVolume(3)->GetCopyNo())*1000;
 if(physVol->GetName() == "TotemT2gem_GEMa27r")UNITA = 1300 + touch->GetVolume(1)->GetCopyNo()+ (touch->GetVolume(3)->GetCopyNo())*1000;
 if(physVol->GetName() == "TotemT2gem_GEMb27r")UNITA = 1400 + touch->GetVolume(1)->GetCopyNo()+ (touch->GetVolume(3)->GetCopyNo())*1000;
 if(physVol->GetName() == "TotemT2gem_GEMc27r")UNITA = 1500 + touch->GetVolume(1)->GetCopyNo()+ (touch->GetVolume(3)->GetCopyNo())*1000;
 if(physVol->GetName() == "TotemT2gem_GAS27r")UNITA = 1600 + touch->GetVolume(1)->GetCopyNo()+ (touch->GetVolume(3)->GetCopyNo())*1000;
 if(physVol->GetName() == "TotemT2gem_strips7r")UNITA = 1700 + touch->GetVolume(1)->GetCopyNo()+ (touch->GetVolume(3)->GetCopyNo())*1000;
 if(physVol->GetName() == "TotemT2gem_isol7r")UNITA = 1800 + touch->GetVolume(1)->GetCopyNo()+ (touch->GetVolume(3)->GetCopyNo())*1000;
 if(physVol->GetName() == "TotemT2gem_pads7r")UNITA = 1900 + touch->GetVolume(1)->GetCopyNo()+ (touch->GetVolume(3)->GetCopyNo())*1000;
 if(physVol->GetName() == "TotemT2gem_HC17r")UNITA = 2000 + touch->GetVolume(1)->GetCopyNo()+ (touch->GetVolume(3)->GetCopyNo())*1000;

 return UNITA;
}
