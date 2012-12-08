#ifndef GEMDigitizer_GEMSimTriv_h
#define GEMDigitizer_GEMSimTriv_h

/** \class GEMSimTriv
 *
 * Class for the GEM strip response simulation based on a very simple model
 *
 * \author Vadim Khotilovich
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

class GEMSimTriv: public GEMSim
{
public:

  GEMSimTriv(const edm::ParameterSet& config);

  ~GEMSimTriv();

  void simulate(const GEMEtaPartition* roll, const edm::PSimHitContainer& rpcHits);

  void simulateNoise(const GEMEtaPartition*);

  void setRandomEngine(CLHEP::HepRandomEngine& eng);

private:

  void init() {}

  GEMSynchronizer* sync_;

  int nbxing_;
  double rate_;
  double gate_;

  //  CLHEP::HepRandomEngine* rndEngine;
  CLHEP::RandFlat* flatDistr1_;
  CLHEP::RandFlat* flatDistr2_;
  CLHEP::RandPoissonQ *poissonDistr_;
};

#endif
