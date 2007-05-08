#ifndef _TRACKER_SIGAUSSIANTAILNOISEADDER_H
#define _TRACKER_SIGAUSSIANTAILNOISEADDER_H

#include "SimTracker/SiStripDigitizer/interface/SiNoiseAdder.h"
#include "SimGeneral/NoiseGenerators/interface/GaussianTailNoiseGenerator.h"

namespace CLHEP {
  class RandGauss;
  class HepRandomEngine;
}

/**
 * Adds the noise only on a subset of strips where it is expected to be greater than a given threshold.
 */
class SiGaussianTailNoiseAdder : public SiNoiseAdder{
 public:
  SiGaussianTailNoiseAdder(int,float,float,CLHEP::HepRandomEngine&);
  virtual ~SiGaussianTailNoiseAdder();
  SiPileUpSignals::signal_map_type addNoise(SiPileUpSignals::signal_map_type);
  void setNumStrips(int in){numStrips = in;}
  void setThreshold(float in){threshold = in;}
 private:
  int numStrips;
  float noiseRMS;
  float threshold;
  CLHEP::RandGauss *gaussDistribution_;
  CLHEP::HepRandomEngine& rndEngine;
};
#endif
