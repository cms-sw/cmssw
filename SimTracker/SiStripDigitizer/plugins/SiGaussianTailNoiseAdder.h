#ifndef _TRACKER_SIGAUSSIANTAILNOISEADDER_H
#define _TRACKER_SIGAUSSIANTAILNOISEADDER_H

#include <memory>

#include "SiNoiseAdder.h"
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
  void addNoise(std::vector<double>&, size_t&, size_t&, int, float) const;
  
  void addNoiseVR(std::vector<double> &, std::vector<float> &) const;
  void addPedestals(std::vector<double> &, std::vector<float> &) const;
  void addCMNoise(std::vector<double> &, float, std::vector<bool> &) const;
  void addBaselineShift(std::vector<double> &, std::vector<bool> &) const;
  
 private:
  const float threshold;
  CLHEP::HepRandomEngine& rndEngine;
  std::unique_ptr<CLHEP::RandGaussQ> gaussDistribution;
  std::unique_ptr<GaussianTailNoiseGenerator> genNoise;
};
#endif
 
