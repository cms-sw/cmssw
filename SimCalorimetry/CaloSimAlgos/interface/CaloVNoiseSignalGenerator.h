#ifndef CaloSimAlgos_CaloVNoiseSignalGenerator_h
#define CaloSimAlgos_CaloVNoiseSignalGenerator_h

#include<vector>
#include "CalibFormats/CaloObjects/interface/CaloSamples.h"

class CaloVNoiseSignalGenerator
{
public:
  CaloVNoiseSignalGenerator();
  virtual ~CaloVNoiseSignalGenerator() {}

  ///  fill theNoiseSignals with one event's worth of noise, in units of pe
  void fillEvent();

  void getNoiseSignals(std::vector<CaloSamples> & noiseSignals) { noiseSignals = theNoiseSignals; }

  bool contains(const DetId & detId) const;

protected:
  virtual void fillNoiseSignals() = 0;
  std::vector<CaloSamples> theNoiseSignals;

private:
  void fillDetIds();
  std::vector<unsigned int> theDetIds;
};

#endif

