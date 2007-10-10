
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
DEFINE_FWK_MODULE(TopMuonSelector);
DEFINE_FWK_MODULE(TopTauSelector);
DEFINE_FWK_MODULE(TopJetSelector);
DEFINE_FWK_MODULE(TopMETSelector);
DEFINE_FWK_MODULE(TopParticleSelector);

#include "TopQuarkAnalysis/TopObjectProducers/interface/TopLeptonCountFilter.h"
#include "TopQuarkAnalysis/TopObjectProducers/interface/TopObjectFilter.h"

DEFINE_FWK_MODULE(TopLeptonCountFilter);
DEFINE_FWK_MODULE(TopElectronCountFilter);
DEFINE_FWK_MODULE(TopMuonCountFilter);
DEFINE_FWK_MODULE(TopTauCountFilter);
DEFINE_FWK_MODULE(TopJetCountFilter);
DEFINE_FWK_MODULE(TopMETCountFilter);
DEFINE_FWK_MODULE(TopParticleCountFilter);

#include "TopQuarkAnalysis/TopObjectProducers/interface/TopObjectEnergyScale.h"

typedef TopObjectEnergyScale<TopElectron> TopElectronEnergyScale;
typedef TopObjectEnergyScale<TopMuon>     TopMuonEnergyScale;
typedef TopObjectEnergyScale<TopTau>      TopTauEnergyScale;
typedef TopObjectEnergyScale<TopJet>      TopJetEnergyScale;
typedef TopObjectEnergyScale<TopMET>      TopMETEnergyScale;

DEFINE_FWK_MODULE(TopElectronEnergyScale);
DEFINE_FWK_MODULE(TopMuonEnergyScale);
DEFINE_FWK_MODULE(TopTauEnergyScale);
DEFINE_FWK_MODULE(TopJetEnergyScale);
DEFINE_FWK_MODULE(TopMETEnergyScale);
