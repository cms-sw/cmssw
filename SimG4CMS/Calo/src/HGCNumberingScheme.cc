///////////////////////////////////////////////////////////////////////////////
// File: HGCNumberingScheme.cc
// Description: Numbering scheme for High Granularity Calorimeter
///////////////////////////////////////////////////////////////////////////////
#include "SimG4CMS/Calo/interface/HGCNumberingScheme.h"
#include "SimDataFormats/CaloTest/interface/HGCalTestNumbering.h"
#include "Geometry/HGCalCommonData/interface/HGCalGeometryMode.h"
#include "DataFormats/Math/interface/FastMath.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/ForwardDetId/interface/HGCalDetId.h"

#include "CLHEP/Units/GlobalSystemOfUnits.h"
#include <iostream>

//#define DebugLog

HGCNumberingScheme::HGCNumberingScheme(const HGCalDDDConstants& hgc, 
				       std::string & name ) :
  hgcons_(hgc) {
  edm::LogInfo("HGCSim") << "Creating HGCNumberingScheme for " << name;
}

HGCNumberingScheme::~HGCNumberingScheme() {
  edm::LogInfo("HGCSim") << "Deleting HGCNumberingScheme";
}

//
uint32_t HGCNumberingScheme::getUnitID(ForwardSubdetector subdet, int layer, 
				       int module, int cell, int iz, 
				       const G4ThreeVector &pos) {
  // module is the sector # for square cell
  //           the copy number of the wafer as placed in the layer
  int      phiSector(0), icell(0), celltyp(0), wafer(0);
  uint32_t index(0);
  if (hgcons_.geomMode() == HGCalGeometryMode::Square) {
    std::pair<int,int> phicell = hgcons_.assignCell(pos.x(),pos.y(),layer,0,false);
    phiSector = phicell.first;
    icell     = phicell.second;
  
    //build the index
    index = HGCalTestNumbering::packSquareIndex(iz,layer,module,phiSector,icell);
    //check if it fits
    if (!hgcons_.isValid(layer,module,icell,false)) {
      index = 0;
    }
  } else if (hgcons_.geomMode() == HGCalGeometryMode::HexagonFull) {
    if (cell >= 0) {
      wafer =  hgcons_.waferFromCopy(module);
      celltyp = cell/1000;
      icell   = cell%1000;
    } else {
      hgcons_.waferFromPosition(pos.x(),pos.y(),wafer,icell,celltyp);
    }
    if (celltyp != 1) celltyp = 0;    
    index   = HGCalTestNumbering::packHexagonIndex((int)subdet,iz,layer,wafer, 
						   celltyp,icell);    
  } else {    
    wafer =  hgcons_.waferFromCopy(module);
    celltyp = cell/1000;
    icell   = cell%1000;
    if (celltyp != 1) celltyp = 0;    
    
    index   = HGCalTestNumbering::packHexagonIndex((int)subdet,iz,layer,wafer, 
						   celltyp,icell);    
    //check if it fits
    if (!hgcons_.isValid(layer,wafer,icell,false)) {
      index = 0;
      edm::LogError("HGCSim") << "[HGCNumberingScheme] ID out of bounds :"
                              << " Subdet= " << subdet << " Zside= " << iz
                              << " Layer= " << layer << " Wafer= " << wafer
                              << " CellType= " << celltyp << " Cell= "
                              << icell;
    }
  }
#ifdef DebugLog
  std::cout << "HGCNumberingScheme::i/p " << subdet << ":" << layer << ":" 
	      << module << ":" << iz << ":";
  if (hgcons_.geomMode() == HGCalGeometryMode::Square) 
    std::cout << pos << " o/p " << phiSector << ":" << icell;
  else
    std::cout << wafer << ":" << celltyp << ":" << icell;
  std::cout << ":" << std::hex << index << std::dec  << std::endl;
#endif
  return index;
}

//
int HGCNumberingScheme::assignCell(float x, float y, int layer) {

  std::pair<int,int> phicell = hgcons_.assignCell(x,y,layer,0,false);
  return phicell.second;
}

//
std::pair<float,float> HGCNumberingScheme::getLocalCoords(int cell, int layer){

  return hgcons_.locateCell(cell,layer,0,false);  
}
