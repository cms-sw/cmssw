
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

DEFINE_FWK_MODULE(TopLeptonCountFilter);

#include "TopQuarkAnalysis/TopObjectProducers/interface/TopObjectFilter.h"

DEFINE_FWK_MODULE(TopElectronMinFilter);
DEFINE_FWK_MODULE(TopMuonMinFilter);
DEFINE_FWK_MODULE(TopTauMinFilter);
DEFINE_FWK_MODULE(TopJetMinFilter);
DEFINE_FWK_MODULE(TopMETMinFilter);
DEFINE_FWK_MODULE(TopParticleMinFilter);

DEFINE_FWK_MODULE(TopElectronMaxFilter);
DEFINE_FWK_MODULE(TopMuonMaxFilter);
DEFINE_FWK_MODULE(TopTauMaxFilter);
DEFINE_FWK_MODULE(TopJetMaxFilter);
DEFINE_FWK_MODULE(TopMETMaxFilter);
DEFINE_FWK_MODULE(TopParticleMaxFilter);


#include "AnalysisDataFormats/TopObjects/interface/TopElectron.h"
#include "AnalysisDataFormats/TopObjects/interface/TopMuon.h"
#include "AnalysisDataFormats/TopObjects/interface/TopTau.h"
#include "AnalysisDataFormats/TopObjects/interface/TopJet.h"
#include "AnalysisDataFormats/TopObjects/interface/TopMET.h"

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

#include "TopQuarkAnalysis/TopObjectProducers/interface/TopObjectSpatialResolution.h"

typedef TopObjectSpatialResolution<TopElectron> TopElectronSpatialResolution;
typedef TopObjectSpatialResolution<TopMuon>     TopMuonSpatialResolution;
typedef TopObjectSpatialResolution<TopTau>      TopTauSpatialResolution;
typedef TopObjectSpatialResolution<TopJet>      TopJetSpatialResolution;
typedef TopObjectSpatialResolution<TopMET>      TopMETSpatialResolution;

DEFINE_FWK_MODULE(TopElectronSpatialResolution);
DEFINE_FWK_MODULE(TopMuonSpatialResolution);
DEFINE_FWK_MODULE(TopTauSpatialResolution);
DEFINE_FWK_MODULE(TopJetSpatialResolution);
DEFINE_FWK_MODULE(TopMETSpatialResolution);
