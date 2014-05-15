#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "SimMuon/GEMDigitizer/interface/GEMDigiModelFactory.h"


#include "SimMuon/GEMDigitizer/interface/GEMDigiProducer.h"
DEFINE_FWK_MODULE(GEMDigiProducer);

#include "SimMuon/GEMDigitizer/interface/GEMTrivialModel.h"
DEFINE_EDM_PLUGIN(GEMDigiModelFactory, GEMTrivialModel, "GEMTrivialModel");

#include "SimMuon/GEMDigitizer/interface/GEMSimpleModel.h"
DEFINE_EDM_PLUGIN(GEMDigiModelFactory, GEMSimpleModel, "GEMSimpleModel");

#include "SimMuon/GEMDigitizer/interface/GEMCSCPadDigiProducer.h"
DEFINE_FWK_MODULE(GEMCSCPadDigiProducer);
