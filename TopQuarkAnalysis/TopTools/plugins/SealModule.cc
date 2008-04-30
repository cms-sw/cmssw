#include "TopQuarkAnalysis/TopTools/plugins/TtJetPartonMatch.h"

typedef TtJetPartonMatch< TtHadEvtPartons  > TtHadEvtJetPartonMatch;
typedef TtJetPartonMatch< TtSemiEvtPartons > TtSemiEvtJetPartonMatch;


#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE(TtHadEvtJetPartonMatch);
DEFINE_FWK_MODULE(TtSemiEvtJetPartonMatch);

