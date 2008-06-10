#include "Validation/RecoJets/plugins/Match.h"
#include "Validation/RecoJets/plugins/Comparison.h"

#include "Validation/RecoJets/plugins/PartonQualifier.h"
#include "Validation/RecoJets/plugins/GenJetQualifier.h"
#include "Validation/RecoJets/plugins/CaloJetQualifier.h"

typedef Comparison<reco::GenJetCollection, GenJetQualifier, Match> GenJetMatch;
typedef Comparison<reco::CaloJetCollection, CaloJetQualifier, Match> CaloJetMatch;
typedef Comparison<reco::GenParticleCollection, PartonQualifier, Match> PartonMatch;
