#ifndef CSCDigitizer_CSCScaNoiseGaussian_h
#define CSCDigitizer_CSCScaNoiseGaussian_h

/** \class CSCScaNoiseGaussian
 * Generate noise for the SCA samples
 * according to a Gaussian distribution
 *
 * \author Rick Wilkinson
 *
 **/
#include<vector>
#include "SimMuon/CSCDigitizer/src/CSCScaNoiseGenerator.h"
#include "CondFormats/CSCObjects/interface/CSCPedestals.h"


class CSCScaNoiseGaussian : public CSCScaNoiseGenerator
{
public:
  CSCScaNoiseGaussian(double analogSignal, double pedestal,
                      double pedestalWidth);
  virtual void noisify(const CSCDetId & layer, CSCAnalogSignal & signal);
  virtual void addPedestal(const CSCDetId & layer, CSCStripDigi & digi);

protected:
  /// default is to do nothing.  Children may read from DB
  virtual void fill(const CSCDetId & layer, int element) {}

  /// all noises are in units of ADC counts
  double analogNoise_;
  double pedestal_;
  double pedestalWidth_;
};

#endif
