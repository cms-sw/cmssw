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
  explicit CSCScaNoiseGaussian(int nScaBins) : CSCScaNoiseGenerator(nScaBins) {}

  virtual CSCPedestals::Item pedestals(const CSCDetId & layer, int element) const = 0;

  /** returns a list of SCA readings  */
  virtual std::vector<int> getNoise(const CSCDetId & layer, int element) const;
};

#endif
