///////////////////////////////////////////////////////////////////////////////
// File: HGCNumberingScheme.cc
// Description: Numbering scheme for High Granularity Calorimeter
///////////////////////////////////////////////////////////////////////////////
#include "SimG4CMS/Calo/interface/HGCNumberingScheme.h"
#include "DataFormats/ForwardDetId/interface/HGCalDetId.h"
#include "DataFormats/Math/interface/FastMath.h"

#include "CLHEP/Units/GlobalSystemOfUnits.h"
#include <iostream>

//#define DebugLog

HGCNumberingScheme::HGCNumberingScheme(const DDCompactView & cpv, 
				       std::string & name, bool check) :
  CaloNumberingScheme(0), check_(check), 
  hgcons(new HGCalDDDConstants(cpv,name)) {
  edm::LogInfo("HGCSim") << "Creating HGCNumberingScheme for " << name;
}

HGCNumberingScheme::~HGCNumberingScheme() {
  edm::LogInfo("HGCSim") << "Deleting HGCNumberingScheme";
}

//
uint32_t HGCNumberingScheme::getUnitID(ForwardSubdetector &subdet, int &layer, int &sector, int &iz, G4ThreeVector &pos) {
  
  std::pair<int,int> phicell = hgcons->assignCell(pos.x(),pos.y(),layer,0,false);
  int phiSector = phicell.first;
  int icell     = phicell.second;
  
  //build the index
  uint32_t index = HGCalDetId(subdet,iz,layer,sector,phiSector,icell).rawId();
  
  //check if it fits
  if ((!HGCalDetId::isValid(subdet,iz,layer,sector,phiSector,icell)) ||
      (!hgcons->isValid(layer,sector,icell,false))) {
    index = 0;
    if (check_) {
      edm::LogError("HGCSim") << "[HGCNumberingScheme] ID out of bounds :"
			      << " Subdet= " << subdet << " Zside= " << iz
			      << " Layer= " << layer << " Sector= " << sector
			      << " SubSector= " << phiSector << " Cell= "
			      << icell << " Local position: (" << pos.x() 
			      << "," << pos.y() << "," << pos.z() << ")";
    }
  }    
#ifdef DebugLog
  std::cout << "HGCNumberingScheme::i/p " << subdet << ":" << layer << ":" 
	    << sector << ":" << iz << ":" << pos << " o/p " << phiSector << ":"
	    << icell << ":" << std::hex << index << std::dec << std::endl;
#endif
  return index;
}

//
int HGCNumberingScheme::assignCell(float x, float y, int layer) {

  std::pair<int,int> phicell = hgcons->assignCell(x,y,layer,0,false);
  return phicell.second;
}

//
std::pair<float,float> HGCNumberingScheme::getLocalCoords(int cell, int layer){

  return hgcons->locateCell(cell,layer,0,false);  
}
