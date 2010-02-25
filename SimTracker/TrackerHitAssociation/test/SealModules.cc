
#include "FWCore/PluginManager/interface/ModuleDef.h"

#include "FWCore/Framework/interface/MakerMacros.h"

#include "SimTracker/TrackerHitAssociation/test/TestAssociator.h"
#include "SimTracker/TrackerHitAssociation/test/myTrackAnalyzer.h"



DEFINE_FWK_MODULE(TestAssociator);
DEFINE_FWK_MODULE(myTrackAnalyzer);


