
#include "PluginManager/ModuleDef.h"

#include "FWCore/Framework/interface/MakerMacros.h"

#include "SimTracker/TrackerHitAssociation/test/TestAssociator.h"

using cms::TestAssociator;

DEFINE_SEAL_MODULE();
DEFINE_ANOTHER_FWK_MODULE(TestAssociator)

