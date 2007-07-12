#ifndef _TRACKER_SIGAUSSIANTAILNOISEADDER_H
#define _TRACKER_SIGAUSSIANTAILNOISEADDER_H

#include "SimTracker/SiStripDigitizer/interface/SiNoiseAdder.h"
#include "SimGeneral/NoiseGenerators/interface/GaussianTailNoiseGenerator.h"

/**
 * Adds the noise only on a subset of strips where it is expected to be greater than a given threshold.
 */

namespace CLHEP {
  class HepRandomEngine;
  class RandGauss;
}

class SiGaussianTailNoiseAdder : public SiNoiseAdder{
 public:
  SiGaussianTailNoiseAdder(float,CLHEP::HepRandomEngine&);
  ~SiGaussianTailNoiseAdder();
  void addNoise(SiPileUpSignals::signal_map_type &,int,float);
  void createRaw(SiPileUpSignals::signal_map_type &,int,float);
 private:
  int numStrips;
  float noiseRMS;
  float threshold;
  CLHEP::HepRandomEngine& rndEngine;
  CLHEP::RandGauss* gaussDistribution;
  GaussianTailNoiseGenerator* genNoise;
};
#endif
