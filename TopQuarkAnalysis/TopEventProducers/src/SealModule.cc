#include "PluginManager/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "TopQuarkAnalysis/TopEventProducers/interface/TtSemiEvtSolutionMaker.h"
#include "TopQuarkAnalysis/TopEventProducers/interface/TtGenEventReco.h"
#include "TopQuarkAnalysis/TopEventProducers/interface/TtDecaySelection.h"

DEFINE_SEAL_MODULE();
DEFINE_ANOTHER_FWK_MODULE(TtGenEventReco);
DEFINE_ANOTHER_FWK_MODULE(TtSemiEvtSolutionMaker);
DEFINE_ANOTHER_FWK_MODULE(TtDecaySelection);
