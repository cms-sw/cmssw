#ifndef CaloSimAlgos_CaloVNoiseSignalGenerator_h
#define CaloSimAlgos_CaloVNoiseSignalGenerator_h

#include "CalibFormats/CaloObjects/interface/CaloSamples.h"
#include <vector>

namespace CLHEP {
  class HepRandomEngine;
}

class CaloVNoiseSignalGenerator {
public:
  CaloVNoiseSignalGenerator();
  virtual ~CaloVNoiseSignalGenerator() {}

  ///  fill theNoiseSignals with one event's worth of noise, in units of pe
  void fillEvent(CLHEP::HepRandomEngine *);

  void fillEvent();  // don't need random engine for some tasks

  void getNoiseSignals(std::vector<CaloSamples> &noiseSignals) { noiseSignals = theNoiseSignals; }

  bool contains(const DetId &detId) const;

  /// if you want to externally fill signals for the event, call this
  /// before fillEvent gets called.
  void setNoiseSignals(const std::vector<CaloSamples> &noiseSignals);

protected:
  /// if you want to fill signals on demand, override this
  /// subclass is responsible for clearing theNoiseSignals before adding
  virtual void fillNoiseSignals(CLHEP::HepRandomEngine *) = 0;
  virtual void fillNoiseSignals() = 0;
  std::vector<CaloSamples> theNoiseSignals;

private:
  void fillDetIds();
  std::vector<unsigned int> theDetIds;
};

#endif
