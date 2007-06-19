#ifndef RPCDigitizer_RPCSimTriv_h
#define RPCDigitizer_RPCSimTriv_h

/** \class RPCSimTriv
 *   Class for the RPC strip response simulation based
 *   on a very simple model
 *
 *  \author Marcello Maggi -- INFN Bari
 */
#include "SimMuon/RPCDigitizer/src/RPCSim.h"


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
			const edm::PSimHitContainer& rpcHits );
 private:
  void init(){};
  CLHEP::HepRandomEngine* rndEngine;
  CLHEP::RandFlat* flatDistribution;

};
#endif
