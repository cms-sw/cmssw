
#include "PluginManager/ModuleDef.h"

#include "FWCore/Framework/interface/MakerMacros.h"

#include "SimTracker/TrackerHitAssociation/test/TestAssociator.h"
#include "SimTracker/TrackerHitAssociation/test/myTrackAnalyzer.h"

using cms::TestAssociator;
using cms::myTrackAnalyzer;

DEFINE_SEAL_MODULE();
DEFINE_ANOTHER_FWK_MODULE(TestAssociator)
DEFINE_ANOTHER_FWK_MODULE(myTrackAnalyzer)


