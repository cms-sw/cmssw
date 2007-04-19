
#include "FWCore/PluginManager/interface/PluginManager.h"

#include "FWCore/Framework/interface/MakerMacros.h"

#include "MixingModule.h"
#include "TestMix.h"

  using edm::MixingModule;
  using edm::TestMix;

DEFINE_SEAL_MODULE();
DEFINE_ANOTHER_FWK_MODULE(MixingModule);
DEFINE_ANOTHER_FWK_MODULE(TestMix);

