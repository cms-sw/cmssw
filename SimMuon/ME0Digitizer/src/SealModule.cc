#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "SimMuon/ME0Digitizer/interface/ME0DigiModelFactory.h"


#include "SimMuon/ME0Digitizer/interface/ME0DigiProducer.h"
DEFINE_FWK_MODULE(ME0DigiProducer);

#include "SimMuon/ME0Digitizer/interface/ME0TrivialModel.h"
DEFINE_EDM_PLUGIN(ME0DigiModelFactory, ME0TrivialModel, "ME0TrivialModel");

#include "SimMuon/ME0Digitizer/interface/ME0SimpleModel.h"
DEFINE_EDM_PLUGIN(ME0DigiModelFactory, ME0SimpleModel, "ME0SimpleModel");
