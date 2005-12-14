///////////////////////////////////////////////////////////////////////////////
// File: CastorNumberingScheme.cc
// Description: Numbering scheme for Castor
///////////////////////////////////////////////////////////////////////////////
#include "SimG4CMS/Forward/interface/CastorNumberingScheme.h"

#include "CLHEP/Units/SystemOfUnits.h"
#include <iostream>

#define debug

CastorNumberingScheme::CastorNumberingScheme(int iv) : CaloNumberingScheme(iv){
  if (verbosity>0) 
    std::cout << "Creating CastorNumberingScheme" << std::endl;
}

CastorNumberingScheme::~CastorNumberingScheme() {
  if (verbosity>0) 
    std::cout << "Deleting CastorNumberingScheme" << std::endl;
}

uint32_t CastorNumberingScheme::getUnitID(const G4Step* aStep) const {

  uint32_t intindex = 0;
  int      level = detectorLevel(aStep);

#ifdef debug
  if (verbosity>2) 
    std::cout << "CastorNumberingScheme number of levels= " << level 
	      << std::endl;
#endif

  if (level > 0) {
    int*      copyno = new int[level];
    G4String* name   = new G4String[level];
    detectorLevel(aStep, level, copyno, name);

    int zside   = 0;
    int sector  = 0;
    int zmodule = 0;
    for (int ich=0; ich  <  level; ich++) {
      if(name[ich] == "CAST") {
	// Z index +Z = 1 ; -Z = 2
	zside   = copyno[ich];
      } else if(name[ich] == "CASS" || name[ich] == "CADS") {
	// sector number for dead material 1 - 8
	sector   = copyno[ich];
      } else if(name[ich] == "CASR" || name[ich] == "CADR") {
	// zmodule number  1 - 18
	zmodule   = copyno[ich];
      } else if(name[ich] == "C3TF") {
	// sector number for sensitive material 1 - 16
	sector   = sector*2 - 1 ;
      } else if(name[ich] == "C4TF") {
	// sector number for sensitive material 1 - 16
	sector   = sector*2 ;
      }
#ifdef debug
      if (verbosity>2)
	std::cout << "CastorNumberingScheme  " << "ich=" << ich  << "copyno" 
		  << copyno[ich] << "name="  << name[ich] << std::endl;
#endif
     }
    // use for Castor number 9 
    // 
    // Z index +Z = 1 ; -Z = 2
    // sector number 1 - 16
    // zmodule number  1 - 18

    int det = 9; 
    intindex = packIndex (det, zside, sector, zmodule);
    

#ifdef debug
    if (verbosity>1) 
      std::cout << "CastorNumberingScheme : det " << det << " zside " 
		<< zside << " sector " << sector << " zmodule " << zmodule
		<< " UnitID 0x" << std::hex << intindex << std::dec 
		<< std::endl;

#endif

    delete[] copyno;
    delete[] name;
  }
  return intindex;
  
}

uint32_t CastorNumberingScheme::packIndex(int det, int z, int sector, 
					  int zmodule ) {
  uint32_t idx=(det&31)<<28;         //bits 28-31   (21-27 are free for now)
  idx+=((z-1)&1)<<20;                //bits  20  (1...2)
  idx+=(sector&15)<<6;               //bits  6-9 (1...16)
  idx+=(zmodule&63);                 //bits  0-5 (1...18)
  return idx;
}

void CastorNumberingScheme::unpackIndex(const uint32_t& idx, int& det, int& z,
					int& sector, int& zmodule) {

  det = (idx>>28)&31;
  z   = (idx>>20)&1;
  z  += 1;
  sector = (idx>>6)&15;
  zmodule= (idx&63);
}
