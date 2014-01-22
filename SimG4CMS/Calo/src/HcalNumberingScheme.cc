///////////////////////////////////////////////////////////////////////////////
// File: HcalNumberingScheme.cc
// Description: Numbering scheme for barrel electromagnetic calorimeter
///////////////////////////////////////////////////////////////////////////////
#include "SimG4CMS/Calo/interface/HcalNumberingScheme.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "DataFormats/HcalDetId/interface/HcalSubdetector.h"

#include "CLHEP/Units/GlobalSystemOfUnits.h"
#include <iostream>

//#define DebugLog

HcalNumberingScheme::HcalNumberingScheme() : CaloNumberingScheme(0) {
  edm::LogInfo("HcalSim") << "Creating HcalNumberingScheme";
}

HcalNumberingScheme::~HcalNumberingScheme() {
  edm::LogInfo("HcalSim") << "Deleting HcalNumberingScheme";
}

uint32_t HcalNumberingScheme::getUnitID(const HcalNumberingFromDDD::HcalID id){

  int zside = 2*(id.zside) - 1;
  int etaR  = zside*(id.etaR);
  HcalSubdetector subdet =  (HcalSubdetector)(id.subdet);

  //pack it into an integer
  // to be consistent with HcalDetId convention
  uint32_t index = HcalDetId(subdet,etaR,id.phis,id.depth).rawId();
#ifdef DebugLog
  edm::LogInfo("HcalSim") << "HcalNumberingScheme det = " << id.subdet 
			  << " depth/lay = " << id.depth << "/" << id.lay 
			  << " zside = " << id.zside << " eta/R = " << id.etaR 
			  << " phi = " << id.phis << " oldphi = " << id.phi
			  << " packed index = 0x" << std::hex << index 
			  << std::dec;
#endif
  return index;

}
