///////////////////////////////////////////////////////////////////////////////
// File: HGCNumberingScheme.cc
// Description: Numbering scheme for High Granularity Calorimeter
///////////////////////////////////////////////////////////////////////////////
#include "SimG4CMS/Calo/interface/HGCNumberingScheme.h"
#include "DataFormats/ForwardDetId/interface/HGCEEDetId.h"
#include "DataFormats/ForwardDetId/interface/HGCHEDetId.h"
#include "DataFormats/ForwardDetId/interface/ForwardSubdetector.h"

#include "CLHEP/Units/GlobalSystemOfUnits.h"
#include <iostream>

#define DebugLog

HGCNumberingScheme::HGCNumberingScheme(std::vector<double> gp) :
  CaloNumberingScheme(0), gpar(gp) {
  edm::LogInfo("HGCSim") << "Creating HGCNumberingScheme";
}

HGCNumberingScheme::~HGCNumberingScheme() {
  edm::LogInfo("HGCSim") << "Deleting HGCNumberingScheme";
}

uint32_t HGCNumberingScheme::getUnitID(int det, G4ThreeVector point, int iz,
				       int module, int layer) {

  int cellx(0), celly(0);
  ForwardSubdetector subdet = (det == 1) ? ForwardSubdetector::HGCEE : ForwardSubdetector::HGCHE;

  //pack it into an integer
  // to be consistent with HGCDetId convention
  uint32_t index = (det == 1) ? HGCEEDetId(subdet,iz,module,layer,cellx).rawId() : HGCHEDetId(subdet,iz,module,layer,cellx,celly).rawId();
#ifdef DebugLog
  edm::LogInfo("HGCSim") << "HGCNumberingScheme det = " << subdet 
			 << " module = " << module << " layer = " << layer
			 << " zside = " << iz << " Cell = " << cellx << ":" 
			 << celly << " packed index = 0x" << std::hex << index 
			 << std::dec;
#endif
  return index;

}
