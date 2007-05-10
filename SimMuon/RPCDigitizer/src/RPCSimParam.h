#ifndef RPCDigitizer_RPCSimParam_h
#define RPCDigitizer_RPCSimParam_h

/** \class RPCSimParam
 *   Class for the RPC strip response simulation based
 *   on a parametrized model (ORCA-based)
 *
 *  \author Marcello Maggi -- INFN Bari
 */
#include "SimMuon/RPCDigitizer/src/RPCSim.h"

namespace CLHEP {
  class HepRandomEngine;
  class RandFlat;
}
class RPCSimParam : public RPCSim
{
 public:
  RPCSimParam(const edm::ParameterSet& config);
  ~RPCSimParam(){}
  void simulate(const RPCRoll* roll,
			const edm::PSimHitContainer& rpcHits );
 private:
  void init(){};
 private:
  double aveEff;
  double aveCls;
  double resRPC;
  double timOff;
  double dtimCs;
  double resEle;
  double sspeed;
  double lbGate;
  bool rpcdigiprint;
  CLHEP::HepRandomEngine* rndEngine;
  CLHEP::RandFlat* flatDistribution;
};
#endif
