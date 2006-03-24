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
#include "DataFormats/CSCDigi/interface/CSCStripDigi.h"
#include "SimMuon/CSCDigitizer/src/CSCAnalogSignal.h"

class CSCScaNoiseGenerator
{
public:
  virtual void noisify(const CSCDetId & layer, CSCAnalogSignal & signal) = 0;
  virtual void addPedestal(const CSCDetId & layer, CSCStripDigi & digi) = 0;
};

#endif
