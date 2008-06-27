#include "PluginManager/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "Matching_fwd.h"
#include "Balancing_fwd.h"
#include "CalibAnalyzer.h"

typedef CalibAnalyzer<reco::GenJetCollection, reco::CaloJetCollection, GenJetCaloJetMatch> GenJetClosure;
typedef CalibAnalyzer<reco::CaloJetCollection, reco::CaloJetCollection, CaloJetCaloJetMatch> CompareCalibs;
typedef CalibAnalyzer<reco::GenParticleCollection, reco::CaloJetCollection, PartonCaloJetMatch> PartonClosure;
typedef CalibAnalyzer<reco::GenParticleCollection, reco::GenJetCollection, PartonGenJetMatch> PartonCorrection;

typedef CalibAnalyzer<reco::PhotonCollection, reco::CaloJetCollection, PhotonCaloJetBalance> PhotonJetClosure;

DEFINE_SEAL_MODULE();
DEFINE_ANOTHER_FWK_MODULE( CompareCalibs );
DEFINE_ANOTHER_FWK_MODULE( GenJetClosure );
DEFINE_ANOTHER_FWK_MODULE( PartonClosure );
DEFINE_ANOTHER_FWK_MODULE( PartonCorrection );
DEFINE_ANOTHER_FWK_MODULE( PhotonJetClosure );


#include "PFJetTester.h"
#include "CaloJetTester.h"

DEFINE_ANOTHER_FWK_MODULE( PFJetTester );
DEFINE_ANOTHER_FWK_MODULE( CaloJetTester );
