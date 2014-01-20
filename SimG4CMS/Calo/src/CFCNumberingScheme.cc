///////////////////////////////////////////////////////////////////////////////
// File: CFCNumberingScheme.cc
// Description: Numbering scheme for Combined Forward Calorimeter
///////////////////////////////////////////////////////////////////////////////
#include "SimG4CMS/Calo/interface/CFCNumberingScheme.h"
#include "DataFormats/ForwardDetId/interface/CFCDetId.h"
#include "DataFormats/ForwardDetId/interface/ForwardSubdetector.h"

#include "CLHEP/Units/GlobalSystemOfUnits.h"
#include <iostream>

#define DebugLog

CFCNumberingScheme::CFCNumberingScheme(std::vector<double> rv, 
				       std::vector<double> xv, 
				       std::vector<double> yv) : 
  CaloNumberingScheme(0), rTable(rv), xCellSize(xv), yCellSize(yv) {
  edm::LogInfo("CFCSim") << "Creating CFCNumberingScheme";
  int ntot(0);
  for (unsigned int i=0; i<xCellSize.size(); ++i) {
    int nmax = (int)((rTable[i+1]-rTable[i]+0.001)/xCellSize[i]);
    ntot    += nmax;
    nMaxX.push_back(ntot);
    std::cout << "CFCNumberingScheme: Range[" << i << "] R = (" << rTable[i] << ":" << rTable[i+1] << " xCell = " << xCellSize[i] << " yCell = " << yCellSize[i] << " nMax " << nMaxX[i] << std::endl;
  }
}

CFCNumberingScheme::~CFCNumberingScheme() {
  edm::LogInfo("CFCSim") << "Deleting CFCNumberingScheme";
}

uint32_t CFCNumberingScheme::getUnitID(G4ThreeVector point, int iz,
				       int module, int fibType, int depth) {

  int nx = nMaxX[xCellSize.size()-1];
  int ny(0);
  for (unsigned int i=0; i<xCellSize.size(); ++i) {
    double r = std::abs(point.x());
    if (r < rTable[i+1]) {
      nx = (int)((r-rTable[i]+0.001)/xCellSize[i]) + 1;
      if (nx < 1) nx  = 1;
      if (i  > 0) nx += nMaxX[i-1];
      ny = (int)((std::abs(point.y())+0.01)/yCellSize[i]) + 1;
      if (point.y() < 0) ny = -ny;
      break;
    }
  }
  ForwardSubdetector subdet =  ForwardSubdetector::CFC;

  //pack it into an integer
  // to be consistent with CFCDetId convention
  uint32_t index = CFCDetId(subdet,module,iz*nx,ny,depth,fibType).rawId();
  std::cout << "CFCNumberingScheme det = " << subdet << " module = " << module << " depth = " << depth << " zside = " << iz << " eta = " << nx << " phi = " << ny << " type = " << fibType << " packed index = 0x" << std::hex << index << std::dec << CFCDetId(index) << std::endl;
#ifdef DebugLog
  edm::LogInfo("CFCSim") << "CFCNumberingScheme det = " << subdet 
			 << " module = " << module << " depth = " << depth 
			 << " zside = " << iz << " eta = " << nx << " phi = " 
			 << ny << " type = " << fibType << " packed index = 0x"
			 << std::hex << index << std::dec << CFCDetId(index);
#endif
  return index;

}
