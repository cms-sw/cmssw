#include "PluginManager/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "Validation/RecoJets/plugins/Matching_fwd.h"
#include "Validation/RecoJets/plugins/Balancing_fwd.h"
#include "Validation/RecoJets/plugins/CalibAnalyzer.h"

typedef CalibAnalyzer<reco::GenJetCollection, GenJetMatch> GenJetClosure;
typedef CalibAnalyzer<reco::CaloJetCollection, CaloJetMatch> CompareCalibs;
typedef CalibAnalyzer<reco::GenParticleCollection, PartonMatch> PartonClosure;

typedef CalibAnalyzer<reco::PhotonCollection, PhotonJetBalance> PhotonJetClosure;

DEFINE_SEAL_MODULE();
DEFINE_ANOTHER_FWK_MODULE( CompareCalibs );
DEFINE_ANOTHER_FWK_MODULE( GenJetClosure );
DEFINE_ANOTHER_FWK_MODULE( PartonClosure );
DEFINE_ANOTHER_FWK_MODULE( PhotonJetClosure );

#include "Validation/RecoJets/plugins/PFJetTester.h"
#include "Validation/RecoJets/plugins/CaloJetTester.h"

DEFINE_ANOTHER_FWK_MODULE( PFJetTester   );
DEFINE_ANOTHER_FWK_MODULE( CaloJetTester );
