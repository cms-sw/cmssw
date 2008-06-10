#include "Validation/RecoJets/plugins/Balance.h"
#include "Validation/RecoJets/plugins/Comparison.h"
#include "Validation/RecoJets/plugins/PhotonQualifier.h"
#include "DataFormats/EgammaCandidates/interface/PhotonFwd.h"

typedef Comparison<reco::PhotonCollection, PhotonQualifier, Balance> PhotonJetBalance;
