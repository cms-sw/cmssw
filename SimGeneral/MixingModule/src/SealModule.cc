
#include "PluginManager/PluginManager.h"

#include "FWCore/Framework/interface/MakerMacros.h"

#include "SimGeneral/MixingModule/interface/MixingModule.h"
#include "SimGeneral/MixingModule/interface/TestMix.h"
#include "SimGeneral/MixingModule/interface/TestSuite.h"

  using edm::MixingModule;
  using edm::TestMix;
  using edm::TestSuite;

  DEFINE_SEAL_MODULE();
  DEFINE_ANOTHER_FWK_MODULE(MixingModule)
  DEFINE_ANOTHER_FWK_MODULE(TestMix)
  DEFINE_ANOTHER_FWK_MODULE(TestSuite)
