#include "FWCore/Framework/interface/MakerMacros.h"

#include "TopQuarkAnalysis/TopTools/plugins/TtJetPartonMatch.h"
#include "AnalysisDataFormats/TopObjects/interface/TtEventPartons.h"

typedef TtJetPartonMatch< TtFullLepEventPartons > TtFullLepJetPartonMatch;
DEFINE_FWK_MODULE(TtFullLepJetPartonMatch);
