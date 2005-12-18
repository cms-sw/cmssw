///////////////////////////////////////////////////////////////////////////////
// File: TBHcalNumberingScheme.cc
// Description: Numbering scheme for test beam hadron calorimeter
///////////////////////////////////////////////////////////////////////////////
#include "SimG4CMS/HcalTestBeam/interface/TBHcalNumberingScheme.h"
#include "SimG4CMS/Calo/interface/HcalTestNumberingScheme.h"
#include "DataFormats/HcalDetId/interface/HcalSubdetector.h"

#include <iostream>


uint32_t TBHcalNumberingScheme::getUnitID(const uint32_t idHit,
					  const int mode) {

  int   subdet, zside, group, ieta, iphi, lay;
  HcalTestNumberingScheme::unpackHcalIndex(idHit, subdet, zside, group,
					   ieta, iphi, lay);
  if (verbosity > 2)
    std::cout << "TBHcalNumberingScheme: i/p ID 0x" << std::hex << idHit 
	      << std::dec << " det " << zside << " group/layer " << group 
	      << " " << lay << " eta/phi " << ieta << " " << iphi << std::endl;

  uint32_t idunit;
  if (mode > 0) {
    if (subdet == static_cast<int>(HcalBarrel) && iphi > 4) {
      if (lay <= 17) {
	// HB2 (unmasked and masked)
	if (ieta > 4 && ieta < 10) {
	  idunit = HcalTestNumberingScheme::packHcalIndex(0,0,1,0,iphi,lay);
	} else {
	  idunit=HcalTestNumberingScheme::packHcalIndex(0,0,1,ieta,iphi,group);
	}
      } else {
	// HO behind HB2
	idunit = HcalTestNumberingScheme::packHcalIndex(0,0,1,ieta,iphi,18);
      }
    } else {
      // HB1, HE, HO behind HB1
      idunit = HcalTestNumberingScheme::packHcalIndex(0,0,1,ieta,iphi,group);
    }
  } else {
    idunit = HcalTestNumberingScheme::packHcalIndex(0,0,1,ieta,iphi,group);
  }

  if (verbosity > 1) {
    std::cout << " TBHcalNumberingScheme: idHit 0x" << std::hex << idHit 
	      << " idunit 0x" << idunit << std::dec << std::endl;
    HcalTestNumberingScheme::unpackHcalIndex(idunit, subdet, zside, group,
					     ieta, iphi, lay);
    if (verbosity > 2)
      std::cout << "TBHcalNumberingScheme: o/p ID 0x" << std::hex << idunit 
		<< std::dec << " det " << zside << " group/layer " << group 
		<< " " << lay << " eta/phi " << ieta << " " <<iphi <<std::endl;
  }
  return idunit;
}

std::vector<uint32_t> TBHcalNumberingScheme::getUnitIDs(const int type,
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
	  id = HcalTestNumberingScheme::packHcalIndex(0,0,1,ieta,iphi,1);
	  tmp.push_back(id);
	}
      }
      // HO behind HB1
      for (ieta=1; ieta<16; ieta++) {
	for (iphi=2; iphi<5; iphi++) {
	  id = HcalTestNumberingScheme::packHcalIndex(0,0,1,ieta,iphi,2);
	  tmp.push_back(id);
	}
      }
      // HB2
      for (lay=1; lay<18; lay++) {
	for (iphi=5; iphi<9; iphi++) {
	  id = HcalTestNumberingScheme::packHcalIndex(0,0,1,0,iphi,lay);
	  tmp.push_back(id);
	}
      }
      // HO behind HB2
      lay = 18;
      for (ieta=1; ieta<16; ieta++) {
	for (iphi=5; iphi<8; iphi++) {
	  id = HcalTestNumberingScheme::packHcalIndex(0,0,1,ieta,iphi,lay);
	  tmp.push_back(id);
	}
      }
    } else {
      // HB1 & HB2
      for (ieta=1; ieta<17; ieta++) {
	for (iphi=1; iphi<9; iphi++) {
	  id = HcalTestNumberingScheme::packHcalIndex(0,0,1,ieta,iphi,1);
	  tmp.push_back(id);
	}
      }
      // HO behind HB
      for (ieta=1; ieta<16; ieta++) {
	for (iphi=2; iphi<8; iphi++) {
	  id = HcalTestNumberingScheme::packHcalIndex(0,0,1,ieta,iphi,2);
	  tmp.push_back(id);
	}
      }
    }
  }

  if (type > 0) {
    // Include HE id's
    for (ieta=15; ieta<17; ieta++) {
      for (iphi=3; iphi<7; iphi++) {
	id = HcalTestNumberingScheme::packHcalIndex(0,0,1,ieta,iphi,3);
	tmp.push_back(id);
      }
    }
    for (iphi=3; iphi<7; iphi++) {
      id = HcalTestNumberingScheme::packHcalIndex(0,0,1,17,iphi,1);
      tmp.push_back(id);
    }
    for (ieta=18; ieta<21; ieta++) {
      for (iphi=3; iphi<7; iphi++) {
	for (int idep=1; idep<3; idep++) {
	  id = HcalTestNumberingScheme::packHcalIndex(0,0,1,ieta,iphi,idep);
	  tmp.push_back(id);
	}
      }
    }
    for (ieta=21; ieta<26; ieta++) {
      for (iphi=2; iphi<4; iphi++) {
	for (int idep=1; idep<3; idep++) {
	  id = HcalTestNumberingScheme::packHcalIndex(0,0,1,ieta,iphi,idep);
	  tmp.push_back(id);
	}
      }
    }
  }

  return tmp;
}
