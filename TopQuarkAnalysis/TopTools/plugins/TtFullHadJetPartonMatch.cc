#include "FWCore/Framework/interface/MakerMacros.h"

#include "TopQuarkAnalysis/TopTools/plugins/TtJetPartonMatch.h"
#include "AnalysisDataFormats/TopObjects/interface/TtFullHadEvtPartons.h"

typedef TtJetPartonMatch< TtFullHadEvtPartons > TtFullHadJetPartonMatch;
DEFINE_FWK_MODULE(TtFullHadJetPartonMatch);
