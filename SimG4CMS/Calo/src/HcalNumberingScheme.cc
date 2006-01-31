///////////////////////////////////////////////////////////////////////////////
// File: HcalNumberingScheme.cc
// Description: Numbering scheme for barrel electromagnetic calorimeter
///////////////////////////////////////////////////////////////////////////////
#include "SimG4CMS/Calo/interface/HcalNumberingScheme.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "DataFormats/HcalDetId/interface/HcalSubdetector.h"

#include "CLHEP/Units/SystemOfUnits.h"
#include <iostream>

HcalNumberingScheme::HcalNumberingScheme(int iv) : CaloNumberingScheme(iv) {
  if (verbosity>0) std::cout << "Creating HcalNumberingScheme" << std::endl;
}

HcalNumberingScheme::~HcalNumberingScheme() {
  if (verbosity>0) std::cout << "Deleting HcalNumberingScheme" << std::endl;
}

uint32_t HcalNumberingScheme::getUnitID(const HcalNumberingFromDDD::HcalID id){

  int zside = 2*(id.zside) - 1;
  int etaR  = zside*(id.etaR);
  HcalSubdetector subdet =  (HcalSubdetector)(id.subdet);

  //pack it into an integer
  // to be consistent with HcalDetId convention
  uint32_t index = HcalDetId(subdet,etaR,id.phis,id.depth).rawId();

  if (verbosity>1) 
    std::cout << "HcalNumberingScheme det = " << id.subdet << " depth/lay = " 
	      << id.depth << "/" << id.lay << " zside = " << id.zside 
	      << " eta/R = " << id.etaR << " phi = " << id.phis << " oldphi = " << id.phi
	      << " packed index = 0x" << std::hex << index << std::dec 
	      << std::endl;

  return index;

}
