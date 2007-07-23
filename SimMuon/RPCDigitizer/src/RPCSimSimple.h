#ifndef RPCDigitizer_RPCSimSimple_h
#define RPCDigitizer_RPCSimSimple_h

/** \class RPCSimSimple
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

class RPCSimSimple : public RPCSim
{
 public:
  RPCSimSimple(const edm::ParameterSet& config);
  ~RPCSimSimple(){}
  void simulate(const RPCRoll* roll,
		const edm::PSimHitContainer& rpcHits ){};

  void simulate(const RPCRoll* roll,
		const edm::PSimHitContainer& rpcHits, const RPCGeometry*);

 private:
  void init(){};
<<<<<<< RPCSimSimple.h
  CLHEP::HepRandomEngine* rndEngine;
  CLHEP::RandFlat* flatDistribution;
  RPCSynchronizer* _rpcSync;

=======
  CLHEP::HepRandomEngine* rndEngine;
  CLHEP::RandFlat* flatDistribution;

>>>>>>> 1.4
};
#endif
