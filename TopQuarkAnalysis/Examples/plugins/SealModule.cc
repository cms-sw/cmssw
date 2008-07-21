#include "TopQuarkAnalysis/Examples/plugins/HypothesisAnalyzer.h"
#include "TopQuarkAnalysis/Examples/plugins/TopMuonAnalyzer.h"
#include "TopQuarkAnalysis/Examples/plugins/TopElecAnalyzer.h"
#include "TopQuarkAnalysis/Examples/plugins/TopJetAnalyzer.h"
#include "TopQuarkAnalysis/Examples/interface/TtSemiEvtKit.h"

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(HypothesisAnalyzer);
DEFINE_FWK_MODULE(TopMuonAnalyzer);
DEFINE_FWK_MODULE(TopElecAnalyzer);
DEFINE_FWK_MODULE(TopJetAnalyzer);
DEFINE_FWK_MODULE(TtSemiEvtKit);

