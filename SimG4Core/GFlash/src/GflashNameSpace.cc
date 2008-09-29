#include "SimG4Core/GFlash/interface/GflashNameSpace.h"

namespace Gflash {

  CalorimeterNumber getCalorimeterNumber(const G4ThreeVector position)
  {
    CalorimeterNumber index = kNULL;
    G4double eta = position.getEta();

    //central
    if (fabs(eta) < EtaMax[kESPM] || fabs(eta) < EtaMax[kHB]) {
      if(position.getRho() > Rmin[kESPM] &&
	 position.getRho() < Rmax[kESPM] ) {
	index = kESPM;
      }
      if(position.getRho() > Rmin[kHB] &&
	 position.getRho() < Rmax[kHB]) {
	index = kHB;
      }
    }
    //forward
    else if (fabs(eta) > EtaMin[kENCA] || fabs(eta) > EtaMin[kHE]) {
      if( fabs(position.getZ()) > Zmin[kENCA] &&
	  fabs(position.getZ()) < Zmax[kENCA] ) {
	index = kENCA;
      }
      if( fabs(position.getZ()) > Zmin[kHE] &&
	  fabs(position.getZ()) < Zmax[kHE] ) {
	index = kHE;
      }
    }
    return index;
  }

}
