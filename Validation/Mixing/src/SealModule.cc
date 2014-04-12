
#include "FWCore/PluginManager/interface/PluginManager.h"

#include "FWCore/Framework/interface/MakerMacros.h"

#include "Validation/Mixing/interface/TestSuite.h"
#include "Validation/Mixing/interface/GlobalTest.h"
#include "Validation/Mixing/interface/MixCollectionValidation.h"


DEFINE_FWK_MODULE(TestSuite);
DEFINE_FWK_MODULE(GlobalTest);
DEFINE_FWK_MODULE(MixCollectionValidation);

