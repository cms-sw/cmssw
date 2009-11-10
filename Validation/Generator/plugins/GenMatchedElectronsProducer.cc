#include "DataFormats/HepMCCandidate/interface/GenParticle.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "Validation/Generator/plugins/GenMatchedCandsProducerBase.h"
typedef GenMatchedCandsProducerBase<reco::GsfElectron, reco::GenParticle> GenMatchedElectronsProducer;

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(GenMatchedElectronsProducer);
