
#include "PluginManager/PluginManager.h"

#include "FWCore/Framework/interface/MakerMacros.h"

#include "Validation/Mixing/interface/TestSuite.h"

  using edm::TestSuite;

  DEFINE_SEAL_MODULE();
DEFINE_ANOTHER_FWK_MODULE(TestSuite);
