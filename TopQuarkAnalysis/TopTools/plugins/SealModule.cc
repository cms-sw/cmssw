#include "FWCore/Framework/interface/MakerMacros.h"

#include "TopQuarkAnalysis/TopTools/plugins/TtJetPartonMatch.h"

#include "TopQuarkAnalysis/TopTools/interface/TtFullHadEvtPartons.h"
#include "TopQuarkAnalysis/TopTools/interface/TtFullLepEvtPartons.h"
#include "TopQuarkAnalysis/TopTools/interface/TtSemiLepEvtPartons.h"

typedef TtJetPartonMatch< TtFullHadEvtPartons > TtFullHadJetPartonMatch;
typedef TtJetPartonMatch< TtFullLepEvtPartons > TtFullLepJetPartonMatch;
typedef TtJetPartonMatch< TtSemiLepEvtPartons > TtSemiLepJetPartonMatch;

DEFINE_FWK_MODULE(TtFullHadJetPartonMatch);
DEFINE_FWK_MODULE(TtFullLepJetPartonMatch);
DEFINE_FWK_MODULE(TtSemiLepJetPartonMatch);
