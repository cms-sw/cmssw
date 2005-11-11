///////////////////////////////////////////////////////////////////////////////
// File: HcalTestNumberingScheme.cc
// Description: Numbering scheme for barrel electromagnetic calorimeter
///////////////////////////////////////////////////////////////////////////////
#include "SimG4CMS/Calo/interface/HcalTestNumberingScheme.h"

#include <iostream>

HcalTestNumberingScheme::HcalTestNumberingScheme(int iv) : 
  HcalNumberingScheme(iv) {
  if (verbosity>0) 
    std::cout << "Creating HcalTestNumberingScheme" << std::endl;
}

HcalTestNumberingScheme::~HcalTestNumberingScheme() {
  if (verbosity>0) 
    std::cout << "Deleting HcalTestNumberingScheme" << std::endl;
}

uint32_t HcalTestNumberingScheme::getUnitID(const HcalNumberingFromDDD::HcalID 
					    id) {

  //pack it into an integer
  uint32_t index = packHcalIndex(id.subdet, id.zside, id.depth, id.etaR,
				 id.phi, id.lay);

  if (verbosity>1) 
    std::cout << "HcalTestNumberingScheme det = " << id.subdet 
	      << " depth/lay = " << id.depth << "/" << id.lay << " zside = " 
	      << id.zside << " eta/R = " << id.etaR << " phi = " << id.phi 
	      << " packed index = 0x" << std::hex << index << std::dec 
	      << std::endl;

  return index;

}

uint32_t HcalTestNumberingScheme::packHcalIndex(int det, int z, int depth,
						int eta, int phi, int lay) {

  uint32_t idx=(det&15)<<28;      //bits 28-31 
  idx+=((depth-1)&3)<<26;         //bits 26-27  
  idx+=((lay-1)&31)<<21;          //bits 21-25
  idx+=(z&1)<<20;                 //bits 20
  idx+=(eta&1023)<<10;            //bits 10-19
  idx+=(phi&1023);                //bits  0-9

  return idx;

}

void HcalTestNumberingScheme::unpackHcalIndex(const uint32_t & idx, 
                                              int& det, int& z, 
                                              int& depth, int& eta,
                                              int& phi, int& lay) {
  det  = (idx>>28)&15;
  depth= (idx>>26)&3;  depth+=1;
  lay  = (idx>>21)&31; lay+=1;
  z    = (idx>>20)&1;
  eta  = (idx>>10)&1023;
  phi  = (idx&1023);

}
