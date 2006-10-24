
#include "PluginManager/ModuleDef.h"

#include "FWCore/Framework/interface/MakerMacros.h"

#include "SimTracker/TrackerHitAssociation/test/TestAssociator.h"
#include "SimTracker/TrackerHitAssociation/test/myTrackAnalyzer.h"


DEFINE_SEAL_MODULE();
DEFINE_ANOTHER_FWK_MODULE(TestAssociator);
DEFINE_ANOTHER_FWK_MODULE(myTrackAnalyzer);


