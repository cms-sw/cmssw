#include "FWCore/PluginManager/interface/ModuleDef.h"

#include "FWCore/Framework/interface/MakerMacros.h"

#include "TrackingTools/TrackRefitter/plugins/TracksToTrajectories.h"

DEFINE_SEAL_MODULE();
DEFINE_ANOTHER_FWK_MODULE(TracksToTrajectories);
