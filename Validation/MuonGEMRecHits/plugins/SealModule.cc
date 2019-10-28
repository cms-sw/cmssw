#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "Validation/MuonGEMRecHits/plugins/GEMRecHitsValidation.h"
#include "Validation/MuonGEMRecHits/plugins/GEMRecHitTrackMatch.h"
#include "Validation/MuonGEMRecHits/plugins/MuonGEMRecHitsHarvestor.h"
DEFINE_FWK_MODULE(GEMRecHitsValidation);
DEFINE_FWK_MODULE(GEMRecHitTrackMatch);
DEFINE_FWK_MODULE(MuonGEMRecHitsHarvestor);
