#include "SimMuon/RPCDigitizer/src/RPCSimFactory.h"
#include "SimMuon/RPCDigitizer/src/RPCSimSimple.h"

RPCSim* 
RPCSimFactory::rpcSim()
{
  return new RPCSimSimple();
}
