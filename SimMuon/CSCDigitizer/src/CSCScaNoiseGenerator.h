#ifndef CSCDigitizer_CSCScaNoiseGenerator_h
#define CSCDigitizer_CSCScaNoiseGenerator_h

/** \class CSCScaNoiseGenerator
 * Generate noise for the SCA samples
 *
 * \author Rick Wilkinson
 *
 **/
#include<vector>
#include "DataFormats/MuonDetId/interface/CSCDetId.h"

class CSCScaNoiseGenerator
{
public:
  explicit CSCScaNoiseGenerator(int nScaBins) : nScaBins_(nScaBins) {}

  /** returns a list of SCA readings  */
  virtual std::vector<int> getNoise(const CSCDetId & layer, int element) const = 0;

protected:
  int nScaBins_;
};

#endif
