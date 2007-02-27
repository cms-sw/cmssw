
#include "PluginManager/PluginManager.h"

#include "FWCore/Framework/interface/MakerMacros.h"

#include "Validation/Mixing/interface/TestSuite.h"
#include "Validation/Mixing/interface/GlobalTest.h"


DEFINE_SEAL_MODULE();
DEFINE_ANOTHER_FWK_MODULE(TestSuite);
DEFINE_ANOTHER_FWK_MODULE(GlobalTest);

