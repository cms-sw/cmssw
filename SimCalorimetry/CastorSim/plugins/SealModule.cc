#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "SimCalorimetry/CastorSim/plugins/CastorDigiProducer.h"
#include "SimCalorimetry/CastorSim/plugins/CastorHitAnalyzer.h"
#include "SimCalorimetry/CastorSim/plugins/CastorDigiAnalyzer.h"



DEFINE_FWK_MODULE(CastorDigiProducer);
DEFINE_FWK_MODULE(CastorHitAnalyzer);
DEFINE_FWK_MODULE(CastorDigiAnalyzer);

