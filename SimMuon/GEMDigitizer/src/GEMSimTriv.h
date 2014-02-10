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

namespace CLHEP {
  class HepRandomEngine;
}

class GEMSimTriv: public GEMSim
{
public:

  GEMSimTriv(const edm::ParameterSet& config);

  ~GEMSimTriv();

  void simulate(const GEMEtaPartition* roll, const edm::PSimHitContainer& rpcHits,
                CLHEP::HepRandomEngine*) override;

  void simulateNoise(const GEMEtaPartition*,
                     CLHEP::HepRandomEngine*) override;

private:

  void init() {}

  GEMSynchronizer* sync_;

  int nbxing_;
  double rate_;
  double gate_;
};
#endif
