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

namespace CLHEP
{
  class HepRandomEngine;
  class RandFlat;
  class RandPoissonQ;
}

class GEMSimAverage: public GEMSim
{
public:

  GEMSimAverage(const edm::ParameterSet& config);

  ~GEMSimAverage();

  void simulate(const GEMEtaPartition* roll, const edm::PSimHitContainer& rpcHits);

  void simulateNoise(const GEMEtaPartition* roll);

  void setRandomEngine(CLHEP::HepRandomEngine& eng);

private:

  void init() {}

  GEMSynchronizer* sync_;
  double averageEfficiency_;
  double averageShapingTime_;
  double averageNoiseRate_;
  double bxwidth_;
  int minBunch_;
  int maxBunch_;

  //  CLHEP::HepRandomEngine* rndEngine;
  CLHEP::RandFlat* flatDistr1_;
  CLHEP::RandFlat* flatDistr2_;
  CLHEP::RandPoissonQ *poissonDistr_;
};

#endif
