// -*- C++ -*-
//
// Package:     HcalTestBeam
// Class  :     HcalTB02XtalNumberingScheme
//
// Implementation:
//     Numbering scheme for crystal calorimeter in 2002 test beam
//
// Original Author:
//         Created:  Sun 21 10:14:34 CEST 2006
// $Id: HcalTB02XtalNumberingScheme.cc,v 1.1 2006/06/04 13:59:38 sunanda Exp $
//
  
// system include files
  
// user include files
#include "SimG4CMS/HcalTestBeam/interface/HcalTB02XtalNumberingScheme.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

//
// constructors and destructor
//

HcalTB02XtalNumberingScheme::HcalTB02XtalNumberingScheme() : 
  HcalTB02NumberingScheme() {
  edm::LogInfo("HcalTBSim") << "Creating HcalTB02XtalNumberingScheme";
}

HcalTB02XtalNumberingScheme::~HcalTB02XtalNumberingScheme() {
  edm::LogInfo("HcalTBSim") << "Deleting HcalTB02XtalNumberingScheme";
}

//
// member functions
//
 
int HcalTB02XtalNumberingScheme::getUnitID(const G4Step* aStep) const {

  const G4VTouchable* touch = aStep->GetPreStepPoint()->GetTouchable();
  int idx=touch->GetReplicaNumber(0);
  int idl=0;
  if (touch->GetHistoryDepth() > 0) idl = touch->GetReplicaNumber(1);
  int idunit = idl*100 + idx;
  LogDebug("HcalTBSim") << "HcalTB02XtalNumberingScheme:: Row " << idl
			<< " Column " << idl << " idunit = " << idunit;
  return idunit;
}
