#ifndef MU_END_SCA_NOISE_READER_H
#define MU_END_SCA_NOISE_READER_H  

#include "SimMuon/CSCDigitizer/src/CSCScaNoiseGaussian.h"

/**
 * \class CSCScaNoiseReader
 *  Reads SCA noise from a file
 *
 *  \author Rick Wilkinson
 *
 * Last mod: <BR>
 *
*/

class CSCScaNoiseReader : public CSCScaNoiseGaussian {
public:
  CSCScaNoiseReader(double pedestal, double pedestalWidth);
  virtual ~CSCScaNoiseReader();

  /// reads the analog noise from a dark-current source
  virtual void noisify(const CSCDetId & layer, CSCAnalogSignal & signal);

protected: 
  /// does nothing
  virtual void fill(const CSCDetId & layer, int element) {}

  int nStripEvents;
  std::vector<int> theData;
};

#endif

