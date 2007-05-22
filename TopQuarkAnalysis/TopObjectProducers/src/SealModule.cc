#include "PluginManager/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "TopQuarkAnalysis/TopObjectProducers/interface/TopJetProducer.h"
#include "TopQuarkAnalysis/TopObjectProducers/interface/TopMuonProducer.h"
#include "TopQuarkAnalysis/TopObjectProducers/interface/TopElectronProducer.h"
#include "TopQuarkAnalysis/TopObjectProducers/interface/TopMETProducer.h"

DEFINE_SEAL_MODULE();
DEFINE_ANOTHER_FWK_MODULE(TopJetProducer);
DEFINE_ANOTHER_FWK_MODULE(TopMuonProducer);
DEFINE_ANOTHER_FWK_MODULE(TopElectronProducer);
DEFINE_ANOTHER_FWK_MODULE(TopMETProducer);


