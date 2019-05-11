#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "SimMuon/RPCDigitizer/src/RPCDigiProducer.h"
#include "SimMuon/RPCDigitizer/src/RPCandIRPCDigiProducer.h"
#include "SimMuon/RPCDigitizer/src/RPCSimFactory.h"

#include "SimMuon/RPCDigitizer/src/RPCSimTriv.h"
#include "SimMuon/RPCDigitizer/src/RPCSimSimple.h"
#include "SimMuon/RPCDigitizer/src/RPCSimParam.h"
#include "SimMuon/RPCDigitizer/src/RPCSimAverage.h"
#include "SimMuon/RPCDigitizer/src/RPCSimAverageNoise.h"
#include "SimMuon/RPCDigitizer/src/RPCSimAverageNoiseEff.h"
#include "SimMuon/RPCDigitizer/src/RPCSimAverageNoiseEffCls.h"
#include "SimMuon/RPCDigitizer/src/RPCSimAsymmetricCls.h"
#include "SimMuon/RPCDigitizer/src/RPCSimModelTiming.h"
#include "SimMuon/RPCDigitizer/src/RPCNeutronWriter.h"

DEFINE_FWK_MODULE(RPCDigiProducer);
DEFINE_FWK_MODULE(RPCandIRPCDigiProducer);

DEFINE_EDM_PLUGIN(RPCSimFactory, RPCSimAverageNoiseEffCls, "RPCSimAverageNoiseEffCls");
DEFINE_EDM_PLUGIN(RPCSimFactory, RPCSimAsymmetricCls, "RPCSimAsymmetricCls");
DEFINE_EDM_PLUGIN(RPCSimFactory, RPCSimModelTiming, "RPCSimModelTiming");
DEFINE_FWK_MODULE(RPCNeutronWriter);
