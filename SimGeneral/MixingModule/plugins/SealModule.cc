
#include "FWCore/PluginManager/interface/PluginManager.h"

#include "FWCore/Framework/interface/MakerMacros.h"

#include "MixingModule.h"
#include "TestMix.h"
#include "InputAnalyzer.h"
#include "SecSourceAnalyzer.h"

  using edm::MixingModule;
  using edm::TestMix;
  using edm::InputAnalyzer;
  using edm::SecSourceAnalyzer;
 
DEFINE_SEAL_MODULE();
DEFINE_ANOTHER_FWK_MODULE(MixingModule);
DEFINE_ANOTHER_FWK_MODULE(TestMix);
DEFINE_ANOTHER_FWK_MODULE(InputAnalyzer);
DEFINE_ANOTHER_FWK_MODULE(SecSourceAnalyzer);
