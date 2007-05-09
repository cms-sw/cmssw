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
  CSCScaNoiseSimple(double analogNoise, double pedestal, double pedestalWidth)
  : analogNoise_(analogNoise),
    pedestal_(pedestal),
    width_(width)
  {
  }
 
};

#endif
