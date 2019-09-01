#include "SimGeneral/GFlash/interface/GflashNameSpace.h"

namespace Gflash {

  // beginning of the Gflash name space

  CalorimeterNumber getCalorimeterNumber(const Gflash3Vector &position) {
    // return the calorimeter number of sensitive detectors (coarse)

    CalorimeterNumber index = kNULL;
    double eta = position.getEta();

    // central
    if (std::fabs(eta) < EtaMax[kESPM]) {
      double rho = position.getRho();
      double rhoBack = rhoBackEB(position);
      if (rho > Gflash::Rmin[kESPM] && rho < rhoBack) {
        index = kESPM;
      } else if (rho > Rmin[kHB] && rho < Rmax[kHB]) {
        index = kHB;
      }
    }
    // forward
    else if (std::fabs(eta) < EtaMax[kENCA]) {
      double z = std::fabs(position.getZ());
      double zBack = zBackEE(position);
      if (z > Gflash::Zmin[kENCA] && z < zBack) {
        index = kENCA;
      } else if (z > Zmin[kHE] && z < Zmax[kHE]) {
        index = kHE;
      }
      // HF is not in the standard Gflash implementation yet
      //      if( z > Zmin[kHF] && z < Zmax[kHF] ) {
      //	index = kHF;
      //      }
    }

    return index;
  }

  int findShowerType(const Gflash3Vector &position) {
    // type of hadron showers subject to the shower starting point (ssp)
    // showerType = -1 : default (invalid)
    // showerType =  0 : ssp before EBRY (barrel crystal)
    // showerType =  1 : ssp inside EBRY
    // showerType =  2 : ssp after  EBRY before HB
    // showerType =  3 : ssp inside HB
    // showerType =  4 : ssp before EFRY (endcap crystal)
    // showerType =  5 : ssp inside EFRY
    // showerType =  6 : ssp after  EFRY before HE
    // showerType =  7 : ssp inside HE

    int showerType = -1;

    // central
    double eta = position.getEta();
    if (std::fabs(eta) < EtaMax[kESPM]) {
      double rho = position.getRho();
      double rhoBack = rhoBackEB(position);
      if (rho < Gflash::RFrontCrystalEB)
        showerType = 0;
      else if (rho < rhoBack)
        showerType = 1;
      else if (rho < Rmin[kHB])
        showerType = 2;
      else
        showerType = 3;
    }
    // forward
    else if (std::fabs(eta) < EtaMax[Gflash::kENCA]) {
      double z = std::fabs(position.getZ());
      double zBack = zBackEE(position);
      if (z < Gflash::ZFrontCrystalEE)
        showerType = 4;
      else if (z < zBack)
        showerType = 5;
      else if (z < Zmin[kHE])
        showerType = 6;
      else
        showerType = 7;
    }

    return showerType;
  }

  double rhoBackEB(const Gflash3Vector &position) {
    // return (Gflash::RFrontCrystalEB +
    // Gflash::LengthCrystalEB*std::sin(position.getTheta()));
    return (Gflash::RFrontCrystalEB + Gflash::LengthCrystalEB);
  }

  double zBackEE(const Gflash3Vector &position) {
    // return (Gflash::ZFrontCrystalEE +
    // Gflash::LengthCrystalEE*std::fabs(std::cos(position.getTheta())));
    return (Gflash::ZFrontCrystalEE + Gflash::LengthCrystalEE);
  }

  // end of the Gflash name space
}  // namespace Gflash
