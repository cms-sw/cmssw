
#include "FWCore/PluginManager/interface/PluginManager.h"

#include "FWCore/Framework/interface/MakerMacros.h"

#include "MixingModule.h"
#include "TestMix.h"
#include "CFWriter.h"
#include "InputAnalyzer.h"
#include "SecSourceAnalyzer.h"
#include "TestMixedSource.h"
#include "Mixing2DB.h"

  using edm::MixingModule;
  using edm::TestMix;
  using edm::CFWriter;
  using edm::InputAnalyzer;
  using edm::SecSourceAnalyzer;
  using edm::TestMixedSource;
  

DEFINE_FWK_MODULE(MixingModule);
DEFINE_FWK_MODULE(TestMix);
DEFINE_FWK_MODULE(CFWriter);
DEFINE_FWK_MODULE(InputAnalyzer);
DEFINE_FWK_MODULE(SecSourceAnalyzer);
DEFINE_FWK_MODULE(TestMixedSource);
DEFINE_FWK_MODULE(Mixing2DB);
