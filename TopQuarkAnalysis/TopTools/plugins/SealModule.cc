#include "FWCore/Framework/interface/MakerMacros.h"

#include "TopQuarkAnalysis/TopTools/plugins/TtJetPartonMatch.h"
#include "TopQuarkAnalysis/TopTools/interface/TtHadEvtPartons.h"
#include "TopQuarkAnalysis/TopTools/interface/TtSemiEvtPartons.h"

typedef TtJetPartonMatch< TtHadEvtPartons  > TtHadEvtJetPartonMatch;
typedef TtJetPartonMatch< TtSemiEvtPartons > TtSemiEvtJetPartonMatch;

DEFINE_FWK_MODULE(TtHadEvtJetPartonMatch);
DEFINE_FWK_MODULE(TtSemiEvtJetPartonMatch);
