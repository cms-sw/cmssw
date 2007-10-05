#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "SimTracker/TrackerFilters/interface/CosmicTIFTrigFilter.h"
using cms::CosmicTIFTrigFilter;

DEFINE_SEAL_MODULE();
DEFINE_ANOTHER_FWK_MODULE(CosmicTIFTrigFilter);
