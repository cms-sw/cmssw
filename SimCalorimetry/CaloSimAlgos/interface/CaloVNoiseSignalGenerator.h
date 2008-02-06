#ifndef CaloSimAlgos_CaloVNoiseSignalGenerator_h
#define CaloSimAlgos_CaloVNoiseSignalGenerator_h

#include<vector>
#include "CalibFormats/CaloObjects/interface/CaloSamples.h"

class CaloVNoiseSignalGenerator
{
public:
  /// needs to be in units of pe
  virtual void getNoiseSignals(std::vector<CaloSamples> & noiseSignals) = 0;
};

#endif

