#ifndef CSCDigitizer_CSCConfigurableStripConditions_h
#define CSCDigitizer_CSCConfigurableStripConditions_h

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "SimGeneral/NoiseGenerators/interface/CorrelatedNoisifier.h"
#include "SimMuon/CSCDigitizer/src/CSCStripConditions.h"

class CSCConfigurableStripConditions : public CSCStripConditions {
public:
  CSCConfigurableStripConditions(const edm::ParameterSet &p);
  ~CSCConfigurableStripConditions() override;

  /// channels count from 1
  float gain(const CSCDetId &detId, int channel) const override;
  float gainSigma(const CSCDetId &detId, int channel) const override { return theGainSigma; }

  /// in ADC counts
  float pedestal(const CSCDetId &detId, int channel) const override { return thePedestal; }
  float pedestalSigma(const CSCDetId &detId, int channel) const override { return thePedestalSigma; }

  void crosstalk(const CSCDetId &detId,
                 int channel,
                 double stripLength,
                 bool leftRight,
                 float &capacitive,
                 float &resistive) const override;

private:
  void fetchNoisifier(const CSCDetId &detId, int istrip) override;
  void makeNoisifier(int chamberType, const std::vector<double> &correlations);
  std::vector<CSCCorrelatedNoisifier *> theNoisifiers;

  float theGain;
  float theME11Gain;
  float theGainSigma;
  float thePedestal;
  float thePedestalSigma;
  // proportional to slope of neighboring signal, per cm of strip length
  float theCapacitiveCrosstalk;
  // proportional to neighboring signal
  float theResistiveCrosstalk;
};

#endif
