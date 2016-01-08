///////////////////////////////////////////////////////////////////////////////
// File: HGCNumberingScheme.cc
// Description: Numbering scheme for High Granularity Calorimeter
///////////////////////////////////////////////////////////////////////////////
#include "SimG4CMS/Calo/interface/HGCNumberingScheme.h"
#include "SimDataFormats/CaloTest/interface/HGCalTestNumbering.h"
#include "Geometry/HGCalCommonData/interface/HGCalGeometryMode.h"
#include "DataFormats/Math/interface/FastMath.h"

#include "CLHEP/Units/GlobalSystemOfUnits.h"
#include <iostream>

//#define DebugLog

HGCNumberingScheme::HGCNumberingScheme(HGCalDDDConstants* hgc, 
				       std::string & name, bool check,
				       int verbose) :
  CaloNumberingScheme(0), check_(check), verbosity(verbose),
  hgcons(hgc) {
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
  if (hgcons->geomMode() == HGCalGeometryMode::Square) {
    std::pair<int,int> phicell = hgcons->assignCell(pos.x(),pos.y(),layer,0,false);
    phiSector = phicell.first;
    icell     = phicell.second;
  
    //build the index
    index = HGCalTestNumbering::packSquareIndex(iz,layer,module,phiSector,icell);
    //check if it fits
    if (!hgcons->isValid(layer,module,icell,false)) {
      index = 0;
      if (check_ && icell != -1) 
	edm::LogError("HGCSim") << "[HGCNumberingScheme] ID out of bounds :"
				<< " Subdet= " << subdet << " Zside= " << iz
				<< " Layer= " << layer << " Module= " << module
				<< " SubSector= " << phiSector << " Cell= "
				<< icell << " Local position: (" << pos.x() 
				<< "," << pos.y() << "," << pos.z() << ")";
    }
  } else {
    celltyp = cell/1000;
    icell   = cell%1000;
    if (celltyp != 1) celltyp = 0;
    wafer   = hgcons->waferFromCopy(module);
    index   = HGCalTestNumbering::packHexagonIndex((int)subdet,iz,layer,wafer, 
						   celltyp,icell);
    //check if it fits
    if (!hgcons->isValid(layer,wafer,icell,false)) {
      index = 0;
      if (check_) {
	edm::LogError("HGCSim") << "[HGCNumberingScheme] ID out of bounds :"
				<< " Subdet= " << subdet << " Zside= " << iz
				<< " Layer= " << layer << " Wafer= " << module
				<< " CellType= " << celltyp << " Cell= "
				<< icell;
      }    
    }
  }
#ifdef DebugLog
  if (verbosity > 0) {
    std::cout << "HGCNumberingScheme::i/p " << subdet << ":" << layer << ":" 
	      << module << ":" << iz << ":";
    if (hgcons->geomMode() == HGCalGeometryMode::Square) 
      std::cout << pos << " o/p " << phiSector << ":" << icell;
    else
      std::cout << wafer << ":" << celltyp << ":" << icell;
    std::cout << ":" << std::hex << index << std::dec  << std::endl;
  }
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
