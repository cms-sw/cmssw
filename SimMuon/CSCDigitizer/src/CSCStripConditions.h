#ifndef CSCDigitizer_CSCStripConditions_h
#define CSCDigitizer_CSCStripConditions_h

#include "SimGeneral/NoiseGenerators/interface/CorrelatedNoisifier.h"
#include "SimMuon/CSCDigitizer/src/CSCAnalogSignal.h"
#include "DataFormats/MuonDetId/interface/CSCDetId.h"

class CSCStripConditions
{
public:
  CSCStripConditions()
  : theNoisifier(0) {}

  void noisify(const CSCDetId & detId, CSCAnalogSignal & signal);

  /// channels count from 1
  virtual float gain(const CSCDetId & detId, int channel) const = 0;
  virtual float gainVariance(const CSCDetId & detId, int channel) const = 0;
  virtual float smearedGain(const CSCDetId & detId, int channel) const;

  /// in ADC counts
  virtual float pedestal(const CSCDetId & detId, int channel) const = 0;
  
protected:
  virtual void fetchNoisifier(const CSCDetId & detId, int istrip) = 0;

  CorrelatedNoisifier * theNoisifier;
};

#endif

