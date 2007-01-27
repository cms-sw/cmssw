#ifndef CSCDigitizer_CSCStripConditions_h
#define CSCDigitizer_CSCStripConditions_h

#include "CondFormats/CSCObjects/interface/CSCGains.h"
#include "SimGeneral/NoiseGenerators/interface/CorrelatedNoisifier.h"
#include "SimMuon/CSCDigitizer/src/CSCAnalogSignal.h"
#include "DataFormats/MuonDetId/interface/CSCDetId.h"

class CSCStripConditions
{
public:
  CSCStripConditions()
  : theGains(0), theNoisifier(0) {}

  void noisify(const CSCDetId & detId, CSCAnalogSignal & signal);

  
protected:
  virtual void fetchNoisifier(const CSCDetId & detId, int istrip) = 0;

  CSCGains * theGains;
  CorrelatedNoisifier * theNoisifier;
};

#endif

