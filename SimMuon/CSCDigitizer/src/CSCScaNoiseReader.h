#ifndef MU_END_SCA_NOISE_READER_H
#define MU_END_SCA_NOISE_READER_H  

#include "SimMuon/CSCDigitizer/src/CSCScaNoiseGenerator.h"

/**
 * \class CSCScaNoiseReader
 *  Reads SCA noise from a file
 *
 *  \author Rick Wilkinson
 *
 * Last mod: <BR>
 *
*/

class CSCScaNoiseReader : public CSCScaNoiseGenerator {
 public:
  CSCScaNoiseReader();
  virtual ~CSCScaNoiseReader();

  virtual std::vector<int> getNoise() const;

 public:
  int nStripEvents;
  std::vector<int> theData;
};

#endif

