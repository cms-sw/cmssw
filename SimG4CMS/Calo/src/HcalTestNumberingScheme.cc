///////////////////////////////////////////////////////////////////////////////
// File: HcalTestNumberingScheme.cc
// Description: Numbering scheme for barrel electromagnetic calorimeter
///////////////////////////////////////////////////////////////////////////////
#include "SimG4CMS/Calo/interface/HcalTestNumberingScheme.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "DataFormats/HcalDetId/interface/HcalSubdetector.h"

#include <iostream>

HcalTestNumberingScheme::HcalTestNumberingScheme(bool forTB) : 
  HcalNumberingScheme(), forTBH2(forTB) {
  edm::LogInfo("HcalSim") << "Creating HcalTestNumberingScheme with TB Flag "
			  << forTBH2;
}

HcalTestNumberingScheme::~HcalTestNumberingScheme() {
  edm::LogInfo("HcalSim") << "Deleting HcalTestNumberingScheme";
}

uint32_t HcalTestNumberingScheme::getUnitID(const HcalNumberingFromDDD::HcalID 
					    id) {

  //pack it into an integer
  uint32_t index = 0;
  if (forTBH2) {
    // TB H2 Case
    int etaR  = id.etaR;
    int phi   = id.phis;
    HcalSubdetector subdet =  (HcalSubdetector)(id.subdet);
    if (phi > 4) { // HB2 
      if (etaR > 4 && etaR < 10)
	index = HcalDetId(subdet,id.lay,id.phis,1).rawId();
    } else { // HB1
      index = HcalDetId(subdet,etaR,id.phis,id.depth).rawId();
    }
  } else {
    // Test case
    index = packHcalIndex(id.subdet, id.zside, id.depth, id.etaR,
			  id.phi, id.lay);
  }

  LogDebug("HcalSim") << "HcalTestNumberingScheme det = " << id.subdet 
		      << " depth/lay = " << id.depth << "/" << id.lay 
		      << " zside = " << id.zside << " eta/R = " << id.etaR 
		      << " phi = " << id.phi << " packed index = 0x" 
		      << std::hex << index << std::dec;

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
