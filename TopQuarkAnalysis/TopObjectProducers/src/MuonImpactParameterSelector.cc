#include "FWCore/Framework/interface/MakerMacros.h"

#include "DataFormats/PatCandidates/interface/Muon.h"

#include "TopQuarkAnalysis/TopObjectProducers/interface/ImpactParameterSelector.h"

typedef ImpactParameterSelector<pat::Muon> MuonImpactParameterSelector;
DEFINE_FWK_MODULE(MuonImpactParameterSelector);
