#include "PluginManager/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "SimMuon/RPCDigitizer/src/RPCDigiProducer.h"
#include "SimMuon/RPCDigitizer/src/RPCSimFactory.h"
#include "SimMuon/RPCDigitizer/src/RPCSimSimple.h"
#include "SimMuon/RPCDigitizer/src/RPCSimParam.h"
#include "SimMuon/RPCDigitizer/src/RPCSimAverage.h"
#include "SimMuon/RPCDigitizer/src/RPCSimTriv.h"


DEFINE_SEAL_MODULE();
DEFINE_ANOTHER_FWK_MODULE(RPCDigiProducer);

DEFINE_SEAL_PLUGIN(RPCSimFactory,RPCSimSimple,"RPCSimSimple");
DEFINE_SEAL_PLUGIN(RPCSimFactory,RPCSimParam,"RPCSimParam");
DEFINE_SEAL_PLUGIN(RPCSimFactory,RPCSimTriv,"RPCSimAverage");
DEFINE_SEAL_PLUGIN(RPCSimFactory,RPCSimTriv,"RPCSimTriv");
