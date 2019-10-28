#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "Validation/MuonGEMHits/plugins/MuonGEMHitsHarvestor.h"
#include "Validation/MuonGEMHits/plugins/GEMHitsValidation.h"
#include "Validation/MuonGEMHits/plugins/GEMSimTrackMatch.h"
DEFINE_FWK_MODULE(MuonGEMHitsHarvestor);
DEFINE_FWK_MODULE(GEMHitsValidation);
DEFINE_FWK_MODULE(GEMSimTrackMatch);
