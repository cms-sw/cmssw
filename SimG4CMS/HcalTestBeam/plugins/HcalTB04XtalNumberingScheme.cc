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
//

// system include files

// user include files
#include "SimG4CMS/Calo/interface/HcalTestNumberingScheme.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "HcalTB04XtalNumberingScheme.h"

//#define EDM_ML_DEBUG

//
// constructors and destructor
//

HcalTB04XtalNumberingScheme::HcalTB04XtalNumberingScheme() : EcalNumberingScheme() {
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HcalTBSim") << "Creating HcalTB04XtalNumberingScheme";
#endif
}

HcalTB04XtalNumberingScheme::~HcalTB04XtalNumberingScheme() {
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HcalTBSim") << "Deleting HcalTB04XtalNumberingScheme";
#endif
}

//
// member functions
//

uint32_t HcalTB04XtalNumberingScheme::getUnitID(const EcalBaseNumber& baseNumber) const {
  int idx = 0, idl = 0;
  if (baseNumber.getLevels() < 1) {
    edm::LogWarning("HcalTBSim") << "HcalTB04XtalNumberingScheme::getUnitID: "
                                 << "No level found in EcalBaseNumber "
                                 << "Returning 0";
  } else {
    idx = baseNumber.getCopyNumber(0);
    if (baseNumber.getLevels() > 1)
      idl = baseNumber.getCopyNumber(1);
  }
  int det = 10;
  uint32_t idunit = HcalTestNumbering::packHcalIndex(det, 0, 1, idl, idx, 1);

#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HcalTBSim") << "HcalTB04XtalNumberingScheme : Crystal " << idx << " Layer " << idl << " UnitID = 0x"
                                << std::hex << idunit << std::dec;
#endif
  return idunit;
}
