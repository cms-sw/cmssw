// -*- C++ -*-
//
// Package:     HcalTestBeam
// Class  :     HcalTBNumberingScheme
//
// Implementation:
//     Numbering scheme for test beam hadron calorimeter
//
// Original Author:
//         Created:  Tue May 16 10:14:34 CEST 2006
// $Id: HcalTBNumberingScheme.cc,v 1.3 2006/11/13 10:32:15 sunanda Exp $
//
  
// system include files
#include <iostream>

// user include files
#include "SimG4CMS/HcalTestBeam/interface/HcalTBNumberingScheme.h"
#include "SimG4CMS/Calo/interface/HcalTestNumberingScheme.h"
#include "DataFormats/HcalDetId/interface/HcalSubdetector.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

//
// member functions
//

uint32_t HcalTBNumberingScheme::getUnitID(const uint32_t idHit,
					  const int mode) {

  int   subdet, zside, group, ieta, iphi, lay;
  HcalTestNumbering::unpackHcalIndex(idHit, subdet, zside, group,
				     ieta, iphi, lay);
  LogDebug("HcalTBSim") << "HcalTBNumberingScheme: i/p ID 0x" << std::hex
			<< idHit << std::dec << " det " << zside << " group "
			<< group << " layer " << lay << " eta " << ieta 
			<< " phi " << iphi;

  uint32_t idunit;
  if (subdet == static_cast<int>(HcalBarrel)) {
    if (lay <= 17) group = 1;
    else           group = 2;
  }
  if (mode > 0) {
    if (subdet == static_cast<int>(HcalBarrel) && iphi > 4) {
      if (lay <= 17) {
	// HB2 (unmasked and masked)
	if (ieta > 4 && ieta < 10) {
	  idunit = HcalTestNumbering::packHcalIndex(0,0,1,0,iphi,lay);
	} else {
	  idunit = HcalTestNumbering::packHcalIndex(0,0,1,ieta,iphi,group);
	}
      } else {
	// HO behind HB2
	idunit = HcalTestNumbering::packHcalIndex(0,0,1,ieta,iphi,18);
      }
    } else {
      // HB1, HE, HO behind HB1
      idunit = HcalTestNumbering::packHcalIndex(0,0,1,ieta,iphi,group);
    }
  } else {
    idunit = HcalTestNumbering::packHcalIndex(0,0,1,ieta,iphi,group);
  }

  HcalTestNumbering::unpackHcalIndex(idunit, subdet, zside, group,
				     ieta, iphi, lay);
  LogDebug("HcalTBSim") << "HcalTBNumberingScheme: idHit 0x" << std::hex 
			<< idHit << " idunit 0x" << idunit << std::dec << "\n"
			<< "HcalTBNumberingScheme: o/p ID 0x" << std::hex 
			<< idunit << std::dec << " det " << zside << " group " 
			<< group << " layer " << lay << " eta " << ieta 
			<< " phi " << iphi;
  
  return idunit;
}

std::vector<uint32_t> HcalTBNumberingScheme::getUnitIDs(const int type,
							const int mode) {

  std::vector<uint32_t> tmp;
  int      iphi, ieta, lay;
  uint32_t id;

  if (type != 1) {
    // Include HB and HO id's
    if (mode>0) {
      // HB1 and masked part of HB2
      for (ieta=1; ieta<17; ieta++) {
	for (iphi=1; iphi<9; iphi++) {
	  id = HcalTestNumbering::packHcalIndex(0,0,1,ieta,iphi,1);
	  tmp.push_back(id);
	}
      }
      // HO behind HB1
      for (ieta=1; ieta<16; ieta++) {
	for (iphi=2; iphi<5; iphi++) {
	  id = HcalTestNumbering::packHcalIndex(0,0,1,ieta,iphi,2);
	  tmp.push_back(id);
	}
      }
      // HB2
      for (lay=1; lay<18; lay++) {
	for (iphi=5; iphi<9; iphi++) {
	  id = HcalTestNumbering::packHcalIndex(0,0,1,0,iphi,lay);
	  tmp.push_back(id);
	}
      }
      // HO behind HB2
      lay = 18;
      for (ieta=1; ieta<16; ieta++) {
	for (iphi=5; iphi<8; iphi++) {
	  id = HcalTestNumbering::packHcalIndex(0,0,1,ieta,iphi,lay);
	  tmp.push_back(id);
	}
      }
    } else {
      // HB1 & HB2
      for (ieta=1; ieta<17; ieta++) {
	for (iphi=1; iphi<9; iphi++) {
	  id = HcalTestNumbering::packHcalIndex(0,0,1,ieta,iphi,1);
	  tmp.push_back(id);
	}
      }
      // HO behind HB
      for (ieta=1; ieta<16; ieta++) {
	for (iphi=2; iphi<8; iphi++) {
	  id = HcalTestNumbering::packHcalIndex(0,0,1,ieta,iphi,2);
	  tmp.push_back(id);
	}
      }
    }
  }

  if (type > 0) {
    // Include HE id's
    for (ieta=15; ieta<17; ieta++) {
      for (iphi=3; iphi<7; iphi++) {
	id = HcalTestNumbering::packHcalIndex(0,0,1,ieta,iphi,3);
	tmp.push_back(id);
      }
    }
    for (iphi=3; iphi<7; iphi++) {
      id = HcalTestNumbering::packHcalIndex(0,0,1,17,iphi,1);
      tmp.push_back(id);
    }
    for (ieta=18; ieta<21; ieta++) {
      for (iphi=3; iphi<7; iphi++) {
	for (int idep=1; idep<3; idep++) {
	  id = HcalTestNumbering::packHcalIndex(0,0,1,ieta,iphi,idep);
	  tmp.push_back(id);
	}
      }
    }
    for (ieta=21; ieta<26; ieta++) {
      for (iphi=2; iphi<4; iphi++) {
	for (int idep=1; idep<3; idep++) {
	  id = HcalTestNumbering::packHcalIndex(0,0,1,ieta,iphi,idep);
	  tmp.push_back(id);
	}
      }
    }
  }

  return tmp;
}
