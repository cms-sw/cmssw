#ifndef RecoVertex_BeamSpotProducer_BeamSpotWrite2Txt_h
#define RecoVertex_BeamSpotProducer_BeamSpotWrite2Txt_h

#include <fstream>

#include "DataFormats/BeamSpot/interface/BeamSpot.h"

namespace beamspot {

  struct BeamSpotContainer {
    reco::BeamSpot beamspot;
    int run;
    char beginTimeOfFit[32];
    char endTimeOfFit[32];
    int beginLumiOfFit;
    int endLumiOfFit;
    std::time_t reftime[2];
  };

  inline void dumpBeamSpotTxt(std::ofstream& outFile, BeamSpotContainer const& bsContainer) {
    outFile << "Runnumber " << bsContainer.run << std::endl;
    outFile << "BeginTimeOfFit " << bsContainer.beginTimeOfFit << " " << bsContainer.reftime[0] << std::endl;
    outFile << "EndTimeOfFit " << bsContainer.endTimeOfFit << " " << bsContainer.reftime[1] << std::endl;
    outFile << "LumiRange " << bsContainer.beginLumiOfFit << " - " << bsContainer.endLumiOfFit << std::endl;
    outFile << "Type " << bsContainer.beamspot.type() << std::endl;
    outFile << "X0 " << bsContainer.beamspot.x0() << std::endl;
    outFile << "Y0 " << bsContainer.beamspot.y0() << std::endl;
    outFile << "Z0 " << bsContainer.beamspot.z0() << std::endl;
    outFile << "sigmaZ0 " << bsContainer.beamspot.sigmaZ() << std::endl;
    outFile << "dxdz " << bsContainer.beamspot.dxdz() << std::endl;
    outFile << "dydz " << bsContainer.beamspot.dydz() << std::endl;
    outFile << "BeamWidthX " << bsContainer.beamspot.BeamWidthX() << std::endl;
    outFile << "BeamWidthY " << bsContainer.beamspot.BeamWidthY() << std::endl;
    for (int i = 0; i < 6; ++i) {
      outFile << "Cov(" << i << ",j) ";
      for (int j = 0; j < 7; ++j) {
        outFile << bsContainer.beamspot.covariance(i, j) << " ";
      }
      outFile << std::endl;
    }
    // Uncertainties on sigmaX and sigmaY are set to be equal. Legacy from a distant past
    outFile << "Cov(6,j) 0 0 0 0 0 0 " << bsContainer.beamspot.covariance(6, 6) << std::endl;
    outFile << "EmittanceX " << bsContainer.beamspot.emittanceX() << std::endl;
    outFile << "EmittanceY " << bsContainer.beamspot.emittanceY() << std::endl;
    outFile << "BetaStar " << bsContainer.beamspot.betaStar() << std::endl;
  }

}  // end namespace beamspot

#endif
