#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "SimMuon/GEMDigitizer/interface/GEMDigiModelFactory.h"
#include "SimMuon/GEMDigitizer/interface/ME0DigiPreRecoModelFactory.h"
#include "SimMuon/GEMDigitizer/interface/ME0DigiModelFactory.h"

#include "SimMuon/GEMDigitizer/interface/GEMDigiProducer.h"
DEFINE_FWK_MODULE(GEMDigiProducer);

#include "SimMuon/GEMDigitizer/interface/GEMSimpleModel.h"
DEFINE_EDM_PLUGIN(GEMDigiModelFactory, GEMSimpleModel, "GEMSimpleModel");

#include "SimMuon/GEMDigitizer/interface/GEMPadDigiProducer.h"
DEFINE_FWK_MODULE(GEMPadDigiProducer);

#include "SimMuon/GEMDigitizer/interface/GEMPadDigiClusterProducer.h"
DEFINE_FWK_MODULE(GEMPadDigiClusterProducer);

#include "SimMuon/GEMDigitizer/interface/ME0DigiPreRecoProducer.h"
DEFINE_FWK_MODULE(ME0DigiPreRecoProducer);

#include "SimMuon/GEMDigitizer/interface/ME0PreRecoNoSmearModel.h"
DEFINE_EDM_PLUGIN(ME0DigiPreRecoModelFactory, ME0PreRecoNoSmearModel, "ME0PreRecoNoSmearModel");

#include "SimMuon/GEMDigitizer/interface/ME0PreRecoGaussianModel.h"
DEFINE_EDM_PLUGIN(ME0DigiPreRecoModelFactory, ME0PreRecoGaussianModel, "ME0PreRecoGaussianModel");

#include "SimMuon/GEMDigitizer/interface/ME0ReDigiProducer.h"
DEFINE_FWK_MODULE(ME0ReDigiProducer);

#include "SimMuon/GEMDigitizer/interface/ME0DigiProducer.h"
DEFINE_FWK_MODULE(ME0DigiProducer);

#include "SimMuon/GEMDigitizer/interface/ME0SimpleModel.h"
DEFINE_EDM_PLUGIN(ME0DigiModelFactory, ME0SimpleModel, "ME0SimpleModel");

#include "SimMuon/GEMDigitizer/interface/ME0PadDigiProducer.h"
DEFINE_FWK_MODULE(ME0PadDigiProducer);

#include "SimMuon/GEMDigitizer/interface/ME0PadDigiClusterProducer.h"
DEFINE_FWK_MODULE(ME0PadDigiClusterProducer);
