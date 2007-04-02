#ifndef RPCDigitizer_RPCSimTriv_h
#define RPCDigitizer_RPCSimTriv_h

/** \class RPCSimTriv
 *   Class for the RPC strip response simulation based
 *   on a very simple model
 *
 *  \author Marcello Maggi -- INFN Bari
 */
#include "SimMuon/RPCDigitizer/src/RPCSim.h"

class RPCSimTriv : public RPCSim
{
 public:
  RPCSimTriv(const edm::ParameterSet& config);
  ~RPCSimTriv(){}
  void simulate(const RPCRoll* roll,
			const edm::PSimHitContainer& rpcHits );
 private:
  void init(){};
};
#endif
