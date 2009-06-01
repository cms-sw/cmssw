#include "FWCore/Framework/interface/MakerMacros.h"

#include "TopQuarkAnalysis/TopTools/plugins/TtJetPartonMatch.h"
#include "AnalysisDataFormats/TopObjects/interface/TtEventPartons.h"

typedef TtJetPartonMatch< TtSemiLepEventPartons > TtSemiLepJetPartonMatch;
DEFINE_FWK_MODULE(TtSemiLepJetPartonMatch);
