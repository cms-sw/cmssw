#ifndef CSCDigitizer_CSCDbStripConditions_h
#define CSCDigitizer_CSCDbStripConditions_h

#include "SimMuon/CSCDigitizer/src/CSCStripConditions.h"
#include "CondFormats/CSCObjects/interface/CSCNoiseMatrix.h"

class CSCDbStripConditions : public CSCStripConditions
{
public:
  CSCDbStripConditions() : CSCStripConditions() {}


private:
  virtual void fetchNoisifier(const CSCDetId & detId, int istrip);  
  static int dbIndex(const CSCDetId & id);
  CSCNoiseMatrix * theNoiseMatrix;
  
};

#endif


