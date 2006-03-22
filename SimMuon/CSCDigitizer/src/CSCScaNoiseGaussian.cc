#include "SimMuon/CSCDigitizer/src/CSCScaNoiseGaussian.h"
#include "CLHEP/Random/RandGaussQ.h"


std::vector<int> CSCScaNoiseGaussian::getNoise(const CSCDetId & layer, int element) const {
  CSCPedestals::Item item = pedestals(layer, element);
  std::vector<int> result(nScaBins_);
  for(int i = 0; i < nScaBins_; ++i) {
    result[i] = (int)(RandGaussQ::shoot(item.ped, item.rms));
  }
 return result;
}

