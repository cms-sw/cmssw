#include "FWCore/Framework/interface/MakerMacros.h"

#include "TopQuarkAnalysis/TopTools/plugins/TtJetPartonMatch.h"
#include "TopQuarkAnalysis/TopTools/interface/TtFullHadEvtPartons.h"
#include "TopQuarkAnalysis/TopTools/interface/TtSemiLepEvtPartons.h"

typedef TtJetPartonMatch< TtFullHadEvtPartons > TtFullHadJetPartonMatch;
typedef TtJetPartonMatch< TtSemiLepEvtPartons > TtSemiLepJetPartonMatch;

DEFINE_FWK_MODULE(TtFullHadJetPartonMatch);
DEFINE_FWK_MODULE(TtSemiLepJetPartonMatch);
