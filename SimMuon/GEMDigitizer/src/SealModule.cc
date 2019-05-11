#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "SimMuon/GEMDigitizer/interface/ME0DigiPreRecoModelFactory.h"
#include "SimMuon/GEMDigitizer/interface/ME0DigiModelFactory.h"

#include "SimMuon/GEMDigitizer/interface/ME0PreRecoNoSmearModel.h"
DEFINE_EDM_PLUGIN(ME0DigiPreRecoModelFactory, ME0PreRecoNoSmearModel, "ME0PreRecoNoSmearModel");

#include "SimMuon/GEMDigitizer/interface/ME0PreRecoGaussianModel.h"
DEFINE_EDM_PLUGIN(ME0DigiPreRecoModelFactory, ME0PreRecoGaussianModel, "ME0PreRecoGaussianModel");

#include "SimMuon/GEMDigitizer/interface/ME0SimpleModel.h"
DEFINE_EDM_PLUGIN(ME0DigiModelFactory, ME0SimpleModel, "ME0SimpleModel");
