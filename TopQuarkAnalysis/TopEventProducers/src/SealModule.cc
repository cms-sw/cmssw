#include "PluginManager/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "TopQuarkAnalysis/TopEventProducers/interface/TtSemiEvtSolutionMaker.h"
#include "TopQuarkAnalysis/TopEventProducers/interface/TtDilepEvtSolutionMaker.h"
#include "TopQuarkAnalysis/TopEventProducers/interface/StEvtSolutionMaker.h"
#include "TopQuarkAnalysis/TopEventProducers/interface/TtDecaySubset.h"
#include "TopQuarkAnalysis/TopEventProducers/interface/TtGenEventReco.h"
#include "TopQuarkAnalysis/TopEventProducers/interface/StGenEventReco.h"
#include "TopQuarkAnalysis/TopEventProducers/interface/TtDecaySelection.h"

DEFINE_SEAL_MODULE();
DEFINE_ANOTHER_FWK_MODULE(TtDecaySubset);
DEFINE_ANOTHER_FWK_MODULE(TtGenEventReco);
DEFINE_ANOTHER_FWK_MODULE(StGenEventReco);
DEFINE_ANOTHER_FWK_MODULE(TtSemiEvtSolutionMaker);
DEFINE_ANOTHER_FWK_MODULE(TtDilepEvtSolutionMaker);
DEFINE_ANOTHER_FWK_MODULE(StEvtSolutionMaker);
DEFINE_ANOTHER_FWK_MODULE(TtDecaySelection);
