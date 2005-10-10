
#include "PluginManager/PluginManager.h"

#include "FWCore/Framework/interface/MakerMacros.h"

#include "SimGeneral/MixingModule/interface/MixingModule.h"
#include "SimGeneral/MixingModule/interface/TestMix.h"

  using edm::MixingModule;
  using edm::TestMix;

  DEFINE_SEAL_MODULE();
  DEFINE_ANOTHER_FWK_MODULE(MixingModule)
  DEFINE_ANOTHER_FWK_MODULE(TestMix)
