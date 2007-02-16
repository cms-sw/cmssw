#include "PluginManager/ModuleDef.h"

#include "FWCore/Framework/interface/MakerMacros.h"

#include "Validation/RecoMuon/src/MuonTrackAnalyzer.h"
#include "Validation/RecoMuon/src/MuonTrackResidualAnalyzer.h"
#include "Validation/RecoMuon/src/RecoMuonValidator.h"

DEFINE_SEAL_MODULE();
DEFINE_ANOTHER_FWK_MODULE(MuonTrackAnalyzer);
DEFINE_ANOTHER_FWK_MODULE(MuonTrackResidualAnalyzer);
DEFINE_ANOTHER_FWK_MODULE(RecoMuonValidator);
