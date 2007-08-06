#include "FWCore/Framework/interface/MakerMacros.h"

#include "TopQuarkAnalysis/TopObjectProducers/interface/TopElectronProducer.h"
#include "TopQuarkAnalysis/TopObjectProducers/interface/TopMuonProducer.h"
#include "TopQuarkAnalysis/TopObjectProducers/interface/TopTauProducer.h"
#include "TopQuarkAnalysis/TopObjectProducers/interface/TopJetProducer.h"
#include "TopQuarkAnalysis/TopObjectProducers/interface/TopMETProducer.h"


DEFINE_FWK_MODULE(TopElectronProducer);
DEFINE_FWK_MODULE(TopMuonProducer);
DEFINE_FWK_MODULE(TopTauProducer);
DEFINE_FWK_MODULE(TopJetProducer);
DEFINE_FWK_MODULE(TopMETProducer);



#include "TopQuarkAnalysis/TopObjectProducers/interface/TopObjectSelector.h"

DEFINE_FWK_MODULE(CaloJetSelector);
DEFINE_FWK_MODULE(TopElectronSelector);
DEFINE_FWK_MODULE(TopTauSelector);
DEFINE_FWK_MODULE(TopMuonSelector);
DEFINE_FWK_MODULE(TopJetSelector);
DEFINE_FWK_MODULE(TopMETSelector);




