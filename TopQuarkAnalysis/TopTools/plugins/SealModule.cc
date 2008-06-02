#include "FWCore/Framework/interface/MakerMacros.h"

#include "TopQuarkAnalysis/TopTools/plugins/TtJetPartonMatch.h"

typedef TtJetPartonMatch< TtHadEvtPartons  > TtHadEvtJetPartonMatch;
typedef TtJetPartonMatch< TtSemiEvtPartons > TtSemiEvtJetPartonMatch;

DEFINE_FWK_MODULE(TtHadEvtJetPartonMatch);
DEFINE_FWK_MODULE(TtSemiEvtJetPartonMatch);

#include "TopQuarkAnalysis/TopTools/plugins/TtSemiGenMatchHypothesis.h"

DEFINE_FWK_MODULE(TtSemiGenMatchHypothesis);

