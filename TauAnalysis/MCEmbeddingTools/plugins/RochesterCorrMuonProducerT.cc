#include "TauAnalysis/MCEmbeddingTools/plugins/RochesterCorrMuonProducerT.h"

#include "DataFormats/MuonReco/interface/Muon.h" 
#include "DataFormats/PatCandidates/interface/Muon.h"

typedef RochesterCorrMuonProducerT<reco::Muon> RochesterCorrMuonProducer;
typedef RochesterCorrMuonProducerT<pat::Muon> RochesterCorrPATMuonProducer;

#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE(RochesterCorrMuonProducer);
DEFINE_FWK_MODULE(RochesterCorrPATMuonProducer);
