#include "FWCore/Framework/interface/MakerMacros.h"
#include "TopQuarkAnalysis/TopEventProducers/interface/TtSemiEvtSolutionMaker.h"
#include "TopQuarkAnalysis/TopEventProducers/interface/TtDilepEvtSolutionMaker.h"
#include "TopQuarkAnalysis/TopEventProducers/interface/StEvtSolutionMaker.h"
#include "TopQuarkAnalysis/TopEventProducers/interface/TopDecaySubset.h"
#include "TopQuarkAnalysis/TopEventProducers/interface/TtGenEventReco.h"
#include "TopQuarkAnalysis/TopEventProducers/interface/StGenEventReco.h"
#include "TopQuarkAnalysis/TopEventProducers/interface/TtDecaySelection.h"

DEFINE_FWK_MODULE(TopDecaySubset);
DEFINE_FWK_MODULE(TtGenEventReco);
DEFINE_FWK_MODULE(StGenEventReco);
DEFINE_FWK_MODULE(TtSemiEvtSolutionMaker);
DEFINE_FWK_MODULE(TtDilepEvtSolutionMaker);
DEFINE_FWK_MODULE(StEvtSolutionMaker);
DEFINE_FWK_MODULE(TtDecaySelection);
