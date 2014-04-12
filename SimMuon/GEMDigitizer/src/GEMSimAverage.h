#ifndef GEMDigitizer_GEMSimAverage_h
#define GEMDigitizer_GEMSimAverage_h

/** \class GEMSimAverage
 *
 * Class for the GEM strip response simulation that extends the Trivial model
 * with average parameters for the GEM efficiency, timing and noise
 *
 * \author Sven Dildick
 */

#include "SimMuon/GEMDigitizer/src/GEMSim.h"

class GEMGeometry;
class GEMSynchronizer;

namespace CLHEP {
  class HepRandomEngine;
}

class GEMSimAverage: public GEMSim
{
public:

  GEMSimAverage(const edm::ParameterSet& config);

  ~GEMSimAverage();

  void simulate(const GEMEtaPartition* roll, const edm::PSimHitContainer& rpcHits,
                CLHEP::HepRandomEngine*) override;

  void simulateNoise(const GEMEtaPartition* roll,
                     CLHEP::HepRandomEngine*) override;

private:

  void init() {}

  GEMSynchronizer* sync_;
  double averageEfficiency_;
  double averageShapingTime_;
  double averageNoiseRate_;
  double bxwidth_;
  int minBunch_;
  int maxBunch_;
};
#endif
