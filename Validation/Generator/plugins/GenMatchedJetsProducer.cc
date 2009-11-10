#include "DataFormats/JetReco/interface/GenJet.h"
#include "DataFormats/JetReco/interface/CaloJet.h"
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"
#include "Validation/Generator/plugins/GenMatchedCandsProducerBase.h"
typedef GenMatchedCandsProducerBase<reco::CaloJet, reco::GenJet> GenJetMatchedJetsProducer;
typedef GenMatchedCandsProducerBase<reco::CaloJet, reco::GenParticle> PartonMatchedJetsProducer;

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(GenJetMatchedJetsProducer);
DEFINE_FWK_MODULE(PartonMatchedJetsProducer);

