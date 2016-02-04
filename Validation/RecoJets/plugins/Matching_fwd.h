#include "Validation/RecoJets/plugins/Match.h"
#include "Validation/RecoJets/plugins/Comparison.h"

#include "Validation/RecoJets/plugins/PartonQualifier.h"
#include "Validation/RecoJets/plugins/GenJetQualifier.h"
#include "Validation/RecoJets/plugins/CaloJetQualifier.h"

typedef Comparison<reco::GenJetCollection, GenJetQualifier, reco::CaloJetCollection, CaloJetQualifier, Match> GenJetCaloJetMatch;
typedef Comparison<reco::GenParticleCollection, PartonQualifier, reco::GenJetCollection, GenJetQualifier, Match> PartonGenJetMatch;
typedef Comparison<reco::CaloJetCollection, CaloJetQualifier, reco::CaloJetCollection, CaloJetQualifier, Match> CaloJetCaloJetMatch;
typedef Comparison<reco::GenParticleCollection, PartonQualifier, reco::CaloJetCollection, CaloJetQualifier, Match> PartonCaloJetMatch;

