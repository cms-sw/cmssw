// -*- C++ -*-
//
// Package:     HcalTestBeam
// Class  :     HcalTB04XtalNumberingScheme
//
// Implementation:
//     Numbering scheme for crystal calorimeter in 2004 test beam
//
// Original Author:
//         Created:  Tue 16 10:14:34 CEST 2006
// $Id: HcalTB04XtalNumberingScheme.cc,v 1.3 2006/11/13 10:32:15 sunanda Exp $
//
  
// system include files
  
// user include files
#include "SimG4CMS/HcalTestBeam/interface/HcalTB04XtalNumberingScheme.h"
#include "SimG4CMS/Calo/interface/HcalTestNumberingScheme.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

using namespace std;

//
// constructors and destructor
//

HcalTB04XtalNumberingScheme::HcalTB04XtalNumberingScheme() : 
  EcalNumberingScheme() {
  edm::LogInfo("HcalTBSim") << "Creating HcalTB04XtalNumberingScheme";
}

HcalTB04XtalNumberingScheme::~HcalTB04XtalNumberingScheme() {
  edm::LogInfo("HcalTBSim") << "Deleting HcalTB04XtalNumberingScheme";
}

//
// member functions
//

uint32_t HcalTB04XtalNumberingScheme::getUnitID(const EcalBaseNumber& baseNumber) const {

  int idx=0, idl=0;
  if (baseNumber.getLevels()<1) {
    edm::LogWarning("HcalTBSim") << "HcalTB04XtalNumberingScheme::getUnitID: "
				 << "No level found in EcalBaseNumber "
				 << "Returning 0";
  } else {
    idx = baseNumber.getCopyNumber(0);
    if (baseNumber.getLevels() > 1) idl = baseNumber.getCopyNumber(1);
  }
  int  det = 10;
  uint32_t idunit = HcalTestNumbering::packHcalIndex(det,0,1,idl,idx,1);

  LogDebug("HcalTBSim") << "HcalTB04XtalNumberingScheme : Crystal " << idx 
			<< " Layer " << idl << " UnitID = 0x" << std::hex
			<< idunit << std::dec;
  
  return idunit;

}
