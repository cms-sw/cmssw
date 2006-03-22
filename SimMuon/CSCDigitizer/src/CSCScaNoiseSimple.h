#ifndef CSCDigitizer_CSCScaNoiseSimple_h
#define CSCDigitizer_CSCScaNoiseSimple_h

/** \class CSCScaNoiseSimple
 * Generate noise for the SCA samples
 * according to a Gaussian distribution
 * with a given pedestal and width
 *
 * \author Rick Wilkinson
 *
 **/
#include "SimMuon/CSCDigitizer/src/CSCScaNoiseGaussian.h"


class CSCScaNoiseSimple : public CSCScaNoiseGaussian
{
public:
  CSCScaNoiseSimple(int nScaBins, double pedestal, double width)
  : CSCScaNoiseGaussian(nScaBins),
    pedestal_(pedestal),
    width_(width)
  {
  }
 

  virtual CSCPedestals::Item pedestals(const CSCDetId & layer, int element) const {
    CSCPedestals::Item result;
    result.ped = pedestal_;
    result.rms = width_;
    return result;
  }

private:
  double pedestal_;
  double width_;
};

#endif
