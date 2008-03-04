#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "SimMuon/RPCDigitizer/src/RPCDigiProducer.h"
#include "SimMuon/RPCDigitizer/src/RPCSimFactory.h"
#include "SimMuon/RPCDigitizer/src/RPCSimSimple.h"
#include "SimMuon/RPCDigitizer/src/RPCSimParam.h"
#include "SimMuon/RPCDigitizer/src/RPCSimAverage.h"
#include "SimMuon/RPCDigitizer/src/RPCSimTriv.h"


DEFINE_SEAL_MODULE();
DEFINE_ANOTHER_FWK_MODULE(RPCDigiProducer);

DEFINE_EDM_PLUGIN(RPCSimFactory,RPCSimSimple,"RPCSimSimple");
DEFINE_EDM_PLUGIN(RPCSimFactory,RPCSimParam,"RPCSimParam");
DEFINE_EDM_PLUGIN(RPCSimFactory,RPCSimAverage,"RPCSimAverage");
DEFINE_EDM_PLUGIN(RPCSimFactory,RPCSimTriv,"RPCSimTriv");

