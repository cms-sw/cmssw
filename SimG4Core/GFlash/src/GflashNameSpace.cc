#include "SimG4Core/GFlash/interface/GflashNameSpace.h"

namespace Gflash {

  CalorimeterNumber getCalorimeterNumber(const G4ThreeVector position)
  {
    CalorimeterNumber index = kNULL;
    G4double eta = position.getEta();

    //central
    if (fabs(eta) < EtaMax[kESPM] || fabs(eta) < EtaMax[kHB]) {
      G4double rho = position.getRho();
      if(rho > Rmin[kESPM] && rho < Rmax[kESPM] ) {
	index = kESPM;
      }
      if(rho > Rmin[kHB] && rho < Rmax[kHB]) {
	index = kHB;
      }
    }
    //forward
    else if (fabs(eta) > EtaMin[kENCA] || fabs(eta) > EtaMin[kHE]) {
      G4double z = fabs(position.getZ());
      if( z > Zmin[kENCA] && z < Zmax[kENCA] ) {
	index = kENCA;
      }
      if( z > Zmin[kHE] && z < Zmax[kHE] ) {
	index = kHE;
      }
      //HF is not in the standard Gflash implementation yet
      //      if( z > Zmin[kHF] && z < Zmax[kHF] ) {
      //	index = kHF;
      //      }
    }
    return index;
  }

}
