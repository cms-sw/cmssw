#include "FWCore/Framework/interface/MakerMacros.h"

#include "TopQuarkAnalysis/TopTools/plugins/TtJetPartonMatch.h"
#include "AnalysisDataFormats/TopObjects/interface/TtSemiLepEvtPartons.h"

typedef TtJetPartonMatch< TtSemiLepEvtPartons > TtSemiLepJetPartonMatch;
DEFINE_FWK_MODULE(TtSemiLepJetPartonMatch);
