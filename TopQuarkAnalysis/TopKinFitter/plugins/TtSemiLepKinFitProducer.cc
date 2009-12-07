#include "TopQuarkAnalysis/TopKinFitter/plugins/TtSemiLepKinFitProducer.h"

#include "DataFormats/PatCandidates/interface/Muon.h"
typedef TtSemiLepKinFitProducer< std::vector<pat::Muon> > TtSemiLepKinFitProducerMuon;

#include "DataFormats/PatCandidates/interface/Electron.h"
typedef TtSemiLepKinFitProducer< std::vector<pat::Electron> > TtSemiLepKinFitProducerElectron;

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(TtSemiLepKinFitProducerMuon);
DEFINE_FWK_MODULE(TtSemiLepKinFitProducerElectron);
