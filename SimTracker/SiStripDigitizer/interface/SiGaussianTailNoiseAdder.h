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
}

class SiGaussianTailNoiseAdder : public SiNoiseAdder{
 public:
  SiGaussianTailNoiseAdder(float);
  ~SiGaussianTailNoiseAdder();
  void addNoise(std::vector<float>&, size_t&, size_t&, int, float, CLHEP::HepRandomEngine*) const;
  
  void addNoiseVR(std::vector<float> &, std::vector<float> &, CLHEP::HepRandomEngine*) const;
  void addPedestals(std::vector<float> &, std::vector<float> &) const;
  void addCMNoise(std::vector<float> &, float, std::vector<bool> &, CLHEP::HepRandomEngine*) const;
  void addBaselineShift(std::vector<float> &, std::vector<bool> &) const;
  
 private:
  const float threshold;
  std::unique_ptr<GaussianTailNoiseGenerator> genNoise;
};
#endif
