#ifndef CSCDigitizer_CSCConfigurableStripConditions_h
#define CSCDigitizer_CSCConfigurableStripConditions_h

#include "SimMuon/CSCDigitizer/src/CSCStripConditions.h"
#include "SimGeneral/NoiseGenerators/interface/CorrelatedNoisifier.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

class CSCConfigurableStripConditions : public CSCStripConditions
{
public:
  CSCConfigurableStripConditions(const edm::ParameterSet & p);
  virtual ~CSCConfigurableStripConditions();

  /// channels count from 1
  virtual float gain(const CSCDetId & detId, int channel) const;
  virtual float gainVariance(const CSCDetId & detId, int channel) const {return theGainVariance;}

  /// in ADC counts
  virtual float pedestal(const CSCDetId & detId, int channel) const {return thePedestal;}

private:
  virtual void fetchNoisifier(const CSCDetId & detId, int istrip);
  void makeNoisifier(int chamberType, const std::vector<double> & correlations);
  std::vector<CorrelatedNoisifier *> theNoisifiers;
  double theAnalogNoise;

  float theGain;
  float theME11Gain;
  float theGainVariance;
  float thePedestal;

};

#endif


