///////////////////////////////////////////////////////////////////////////////
// File: HGCNumberingScheme.cc
// Description: Numbering scheme for High Granularity Calorimeter
///////////////////////////////////////////////////////////////////////////////
#include "SimG4CMS/Calo/interface/HGCNumberingScheme.h"
#include "DataFormats/ForwardDetId/interface/HGCalDetId.h"
#include "DataFormats/Math/interface/FastMath.h"

#include "CLHEP/Units/GlobalSystemOfUnits.h"
#include <iostream>

#define DebugLog

HGCNumberingScheme::HGCNumberingScheme(const DDCompactView & cpv, 
				       std::string & name) :
  CaloNumberingScheme(0), hgcons(new HGCalDDDConstants(cpv,name)) {
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
  
  //check if it fits
  if (icell>0xffff) {
    edm::LogError("HGCSim") << "[HGCNumberingScheme] cell id seems to be out of bounds cell id=" << icell << "\n"
			    << "\tLocal position: (" << pos.x() << "," << pos.y() << "," << pos.z() << ")\n"
			    << "\tlayer " << layer << "\tsector " << sector;
  }    
  
  //build the index
  uint32_t index = HGCalDetId(subdet,iz,layer,sector,phiSector,icell).rawId();
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
