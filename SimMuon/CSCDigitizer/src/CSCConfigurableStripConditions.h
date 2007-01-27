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

private:
  virtual void fetchNoisifier(const CSCDetId & detId, int istrip);
  void makeNoisifier(int chamberType, const std::vector<double> & correlations);
  std::vector<CorrelatedNoisifier *> theNoisifiers;
  double theAnalogNoise;
};

#endif


