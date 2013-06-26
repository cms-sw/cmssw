#include "FWCore/Framework/interface/MakerMacros.h"
#include "SimCalorimetry/CastorSim/plugins/CastorDigiProducer.h"
#include "SimCalorimetry/CastorSim/plugins/CastorHitAnalyzer.h"
#include "SimCalorimetry/CastorSim/plugins/CastorDigiAnalyzer.h"
#include "SimGeneral/MixingModule/interface/DigiAccumulatorMixModFactory.h"

DEFINE_DIGI_ACCUMULATOR(CastorDigiProducer);
DEFINE_FWK_MODULE(CastorHitAnalyzer);
DEFINE_FWK_MODULE(CastorDigiAnalyzer);

