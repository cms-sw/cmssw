#include "FWCore/Framework/interface/MakerMacros.h"

#include "DataFormats/PatCandidates/interface/Muon.h"
#include "DataFormats/PatCandidates/interface/Electron.h"

#include "TopQuarkAnalysis/TopHitFit/plugins/TtSemiLepHitFitProducer.h"

typedef TtSemiLepHitFitProducer< std::vector<pat::Muon>     > TtSemiLepHitFitProducerMuon;
typedef TtSemiLepHitFitProducer< std::vector<pat::Electron> > TtSemiLepHitFitProducerElectron;

DEFINE_FWK_MODULE(TtSemiLepHitFitProducerMuon);
DEFINE_FWK_MODULE(TtSemiLepHitFitProducerElectron);
