
#include "PluginManager/ModuleDef.h"

#include "FWCore/Framework/interface/MakerMacros.h"

#include "RecoTracker/RoadSearchTrackCandidateMaker/test/ReadTrackCandidates.h"

using cms::ReadTrackCandidates;

DEFINE_SEAL_MODULE();
DEFINE_ANOTHER_FWK_MODULE(ReadTrackCandidates);

