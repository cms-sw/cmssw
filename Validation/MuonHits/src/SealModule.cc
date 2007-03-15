#include "PluginManager/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "Validation/MuonHits/src/MuonSimHitsValidAnalyzer.h"
#include "Validation/MuonHits/interface/MuonSimHitsValidProducer.h"


DEFINE_SEAL_MODULE();
DEFINE_ANOTHER_FWK_MODULE(MuonSimHitsValidAnalyzer);
DEFINE_ANOTHER_FWK_MODULE(MuonSimHitsValidProducer);
