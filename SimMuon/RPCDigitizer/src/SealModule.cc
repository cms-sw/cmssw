#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "SimMuon/RPCDigitizer/src/RPCDigiProducer.h"
#include "SimMuon/RPCDigitizer/src/RPCSimFactory.h"

#include "SimMuon/RPCDigitizer/src/RPCSimTriv.h"
#include "SimMuon/RPCDigitizer/src/RPCSimSimple.h"
#include "SimMuon/RPCDigitizer/src/RPCSimParam.h"
#include "SimMuon/RPCDigitizer/src/RPCSimAverage.h"
#include "SimMuon/RPCDigitizer/src/RPCSimAverageNoise.h"
#include "SimMuon/RPCDigitizer/src/RPCSimAverageNoiseEff.h"
#include "SimMuon/RPCDigitizer/src/RPCSimAverageNoiseEffCls.h"
#include "SimMuon/RPCDigitizer/src/RPCSimAsymmetricCls.h"

#include "SimMuon/RPCDigitizer/src/RPCNeutronWriter.h"

DEFINE_FWK_MODULE(RPCDigiProducer);

DEFINE_EDM_PLUGIN(RPCSimFactory,RPCSimTriv,"RPCSimTriv");
DEFINE_EDM_PLUGIN(RPCSimFactory,RPCSimSimple,"RPCSimSimple");
DEFINE_EDM_PLUGIN(RPCSimFactory,RPCSimParam,"RPCSimParam");
DEFINE_EDM_PLUGIN(RPCSimFactory,RPCSimAverage,"RPCSimAverage");
DEFINE_EDM_PLUGIN(RPCSimFactory,RPCSimAverageNoise,"RPCSimAverageNoise");
DEFINE_EDM_PLUGIN(RPCSimFactory,RPCSimAverageNoiseEff,"RPCSimAverageNoiseEff");
DEFINE_EDM_PLUGIN(RPCSimFactory,RPCSimAverageNoiseEffCls,"RPCSimAverageNoiseEffCls");
DEFINE_EDM_PLUGIN(RPCSimFactory,RPCSimAsymmetricCls,"RPCSimAsymmetricCls");

DEFINE_FWK_MODULE(RPCNeutronWriter);
