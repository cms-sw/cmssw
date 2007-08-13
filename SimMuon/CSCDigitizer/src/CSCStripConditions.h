#ifndef CSCDigitizer_CSCStripConditions_h
#define CSCDigitizer_CSCStripConditions_h

#include "SimGeneral/NoiseGenerators/interface/CorrelatedNoisifier.h"
#include "SimMuon/CSCDigitizer/src/CSCAnalogSignal.h"
#include "DataFormats/MuonDetId/interface/CSCDetId.h"
#include "FWCore/Framework/interface/EventSetup.h"


class CSCStripConditions
{
public:
  CSCStripConditions()
  : theNoisifier(0) {}

  virtual ~CSCStripConditions() {delete theNoisifier;}

  void noisify(const CSCDetId & detId, CSCAnalogSignal & signal);

  virtual void initializeEvent(const edm::EventSetup & es) {}

  /// channels count from 1
  virtual float gain(const CSCDetId & detId, int channel) const = 0;
  virtual float gainVariance(const CSCDetId & detId, int channel) const = 0;
  virtual float smearedGain(const CSCDetId & detId, int channel) const;

  /// in ADC counts
  virtual float pedestal(const CSCDetId & detId, int channel) const = 0;
  virtual float pedestalVariance(const CSCDetId & detId, int channel) const = 0;

  /// calculated from pedestalVariance & gain
  float analogNoise(const CSCDetId & detId, int channel) const;

  virtual void crosstalk(const CSCDetId&detId, int channel,
                 double stripLength, bool leftRight, 
                 float & capacitive, float & resistive) const = 0;
  
protected:
  virtual void fetchNoisifier(const CSCDetId & detId, int istrip) = 0;

  CorrelatedNoisifier * theNoisifier;
};

#endif

