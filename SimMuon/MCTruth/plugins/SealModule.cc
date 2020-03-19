#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "SimMuon/MCTruth/interface/CSCTruthTest.h"
#include "SimMuon/MCTruth/plugins/MuonAssociatorEDProducer.h"
#include "SimMuon/MCTruth/plugins/MuonTrackProducer.h"

DEFINE_FWK_MODULE(MuonAssociatorEDProducer);
DEFINE_FWK_MODULE(MuonTrackProducer);
DEFINE_FWK_MODULE(CSCTruthTest);
