#include "SimMuon/CSCDigitizer/src/CSCScaNoiseGenerator.h"
#include "DataFormats/CSCDigi/interface/CSCStripDigi.h"
#include "CLHEP/Random/RandGaussQ.h"

CSCScaNoiseGenerator::CSCScaNoiseGenerator(int nScaBins)
 : nScaBins_(nScaBins)
{
}


std::vector<int> CSCScaNoiseGenerator::getNoise() const {
  std::vector<int> result(nScaBins_);
  for(int i = 0; i < nScaBins_; ++i) {
    result[i] = (int)(RandGaussQ::shoot() * 0.72);
  }
  return result;
}

