#ifndef _TRACKER_SIGAUSSIANTAILNOISEADDER_H
#define _TRACKER_SIGAUSSIANTAILNOISEADDER_H

#include "SimTracker/SiStripDigitizer/interface/SiNoiseAdder.h"
#include "SimGeneral/NoiseGenerators/interface/GaussianTailNoiseGenerator.h"

/**
 * Adds the noise only on a subset of strips where it is expected to be greater than a given threshold.
 */

namespace CLHEP {
  class HepRandomEngine;
  class RandGaussQ;
}

class SiGaussianTailNoiseAdder : public SiNoiseAdder{
 public:
  SiGaussianTailNoiseAdder(float,CLHEP::HepRandomEngine&);
  ~SiGaussianTailNoiseAdder();
  void addNoise(std::vector<double>&, size_t&, size_t&, int, float);
  //void createRaw(std::vector<double>&, size_t&, size_t&, int, float, float);
  
  void addNoiseVR(std::vector<double> &, std::vector<float> &);
  void addPedestals(std::vector<double> &, std::vector<float> &);
  void addCMNoise(std::vector<double> &, float, std::vector<bool> &);
  void addBaselineShift(std::vector<double> &, std::vector<bool> &);
  
 private:
  //int numStrips;
  //float noiseRMS;
  float pedValue;
  float threshold;
  CLHEP::HepRandomEngine& rndEngine;
  CLHEP::RandGaussQ* gaussDistribution;
  GaussianTailNoiseGenerator* genNoise;
};
#endif
 
