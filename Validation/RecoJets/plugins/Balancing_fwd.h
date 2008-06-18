#include "Validation/RecoJets/plugins/Balance.h"
#include "Validation/RecoJets/plugins/Comparison.h"

#include "Validation/RecoJets/plugins/PhotonQualifier.h"
#include "Validation/RecoJets/plugins/CaloJetQualifier.h"

typedef Comparison<reco::PhotonCollection, PhotonQualifier, reco::CaloJetCollection, CaloJetQualifier, Balance> PhotonCaloJetBalance;
