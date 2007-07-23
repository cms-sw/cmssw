#ifndef RPCDigitizer_RPCSimTriv_h
#define RPCDigitizer_RPCSimTriv_h

/** \class RPCSimTriv
 *   Class for the RPC strip response simulation based
 *   on a very simple model
 *
 *  \author Marcello Maggi -- INFN Bari
 */
#include "SimMuon/RPCDigitizer/src/RPCSim.h"
#include "SimMuon/RPCDigitizer/src/RPCSynchronizer.h"

class RPCGeometry;

namespace CLHEP {
  class HepRandomEngine;
  class RandFlat;
}


namespace CLHEP {
  class HepRandomEngine;
  class RandFlat;
}

class RPCSimTriv : public RPCSim
{
 public:
  RPCSimTriv(const edm::ParameterSet& config);
  ~RPCSimTriv(){}

  void simulate(const RPCRoll* roll,
		const edm::PSimHitContainer& rpcHits){};

  void simulate(const RPCRoll* roll,
		const edm::PSimHitContainer& rpcHits,
		const RPCGeometry* geo );
 private:
  void init(){};
<<<<<<< RPCSimTriv.h
  CLHEP::HepRandomEngine* rndEngine;
  CLHEP::RandFlat* flatDistribution;
  RPCSynchronizer* _rpcSync;

=======
  CLHEP::HepRandomEngine* rndEngine;
  CLHEP::RandFlat* flatDistribution;

>>>>>>> 1.2
};
#endif
