#include "FWCore/Framework/interface/MakerMacros.h"

#include "TopQuarkAnalysis/TopKinFitter/plugins/TtSemiKinFitProducer.h"

#include "DataFormats/PatCandidates/interface/Muon.h"
#include "DataFormats/PatCandidates/interface/Electron.h"

typedef TtSemiKinFitProducer< std::vector<pat::Muon>     > TtSemiKinFitProducerMuon;
typedef TtSemiKinFitProducer< std::vector<pat::Electron> > TtSemiKinFitProducerElectron;

DEFINE_FWK_MODULE(TtSemiKinFitProducerMuon);
DEFINE_FWK_MODULE(TtSemiKinFitProducerElectron);
