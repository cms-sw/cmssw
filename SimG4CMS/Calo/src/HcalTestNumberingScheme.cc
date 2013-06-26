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
    if (subdet == HcalBarrel && phi > 4) { // HB2 
      if (etaR > 4 && etaR < 10)
	index = HcalDetId(subdet,id.lay,id.phis,1).rawId();
    } else { // HB1
      index = HcalDetId(subdet,etaR,id.phis,id.depth).rawId();
    }
  } else {
    // Test case
    index = HcalTestNumbering::packHcalIndex(id.subdet, id.zside, id.depth, 
					     id.etaR, id.phis, id.lay);
  }

  LogDebug("HcalSim") << "HcalTestNumberingScheme det = " << id.subdet 
		      << " depth/lay = " << id.depth << "/" << id.lay 
		      << " zside = " << id.zside << " eta/R = " << id.etaR 
		      << " phi = " << id.phis << " packed index = 0x" 
		      << std::hex << index << std::dec;

  return index;

}

uint32_t HcalTestNumberingScheme::packHcalIndex(int det, int z, int depth, 
						int eta, int phi, int lay) {

  return  HcalTestNumbering::packHcalIndex(det, z, depth, eta, phi, lay);
}

void HcalTestNumberingScheme::unpackHcalIndex(const uint32_t & idx, 
                                              int& det, int& z, 
                                              int& depth, int& eta,
                                              int& phi, int& lay) {

  HcalTestNumbering::unpackHcalIndex(idx, det, z, depth, eta, phi, lay);
}
