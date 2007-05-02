#include "PluginManager/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "TopQuarkAnalysis/TopObjectProducers/interface/TopJetObjectProducer.h"
#include "TopQuarkAnalysis/TopObjectProducers/interface/TopMuonObjectProducer.h"
#include "TopQuarkAnalysis/TopObjectProducers/interface/TopElectronObjectProducer.h"
#include "TopQuarkAnalysis/TopObjectProducers/interface/TopMETObjectProducer.h"

DEFINE_SEAL_MODULE();
DEFINE_ANOTHER_FWK_MODULE(TopJetObjectProducer);
DEFINE_ANOTHER_FWK_MODULE(TopMuonObjectProducer);
DEFINE_ANOTHER_FWK_MODULE(TopElectronObjectProducer);
DEFINE_ANOTHER_FWK_MODULE(TopMETObjectProducer);


