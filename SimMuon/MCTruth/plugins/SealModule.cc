#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "SimMuon/MCTruth/interface/CSCTruthTest.h"
#include "SimMuon/MCTruth/plugins/MuonAssociatorEDProducer.h"
#include "SimMuon/MCTruth/plugins/SeedToTrackProducer.h"
#include "SimMuon/MCTruth/plugins/Phase2SeedToTrackProducer.h"

DEFINE_FWK_MODULE(MuonAssociatorEDProducer);
DEFINE_FWK_MODULE(CSCTruthTest);
DEFINE_FWK_MODULE(SeedToTrackProducer);
DEFINE_FWK_MODULE(Phase2SeedToTrackProducer);
