#include "FWCore/Framework/interface/MakerMacros.h"

#include "DataFormats/PatCandidates/interface/Electron.h"

#include "TopQuarkAnalysis/TopObjectProducers/interface/ImpactParameterSelector.h"

typedef ImpactParameterSelector<pat::Electron> ElectronImpactParameterSelector;
DEFINE_FWK_MODULE(ElectronImpactParameterSelector);
