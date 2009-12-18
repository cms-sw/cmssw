#include "FWCore/Framework/interface/MakerMacros.h"
#include "TopQuarkAnalysis/TopEventProducers/interface/TtSemiEvtSolutionMaker.h"
#include "TopQuarkAnalysis/TopEventProducers/interface/TtDilepEvtSolutionMaker.h"
#include "TopQuarkAnalysis/TopEventProducers/interface/TtHadEvtSolutionMaker.h"
#include "TopQuarkAnalysis/TopEventProducers/interface/StEvtSolutionMaker.h"
#include "TopQuarkAnalysis/TopEventProducers/interface/TopInitSubset.h"
#include "TopQuarkAnalysis/TopEventProducers/interface/TopDecaySubset.h"
#include "TopQuarkAnalysis/TopEventProducers/interface/TtGenEventReco.h"
#include "TopQuarkAnalysis/TopEventProducers/interface/StGenEventReco.h"
#include "TopQuarkAnalysis/TopEventProducers/interface/TtDecaySelection.h"

DEFINE_FWK_MODULE(TopInitSubset);
DEFINE_FWK_MODULE(TopDecaySubset);
DEFINE_FWK_MODULE(TtGenEventReco);
DEFINE_FWK_MODULE(StGenEventReco);
DEFINE_FWK_MODULE(TtSemiEvtSolutionMaker);
DEFINE_FWK_MODULE(TtDilepEvtSolutionMaker);
DEFINE_FWK_MODULE(TtHadEvtSolutionMaker);
DEFINE_FWK_MODULE(StEvtSolutionMaker);
DEFINE_FWK_MODULE(TtDecaySelection);

#include "TopQuarkAnalysis/TopEventProducers/interface/TtEvtBuilder.h"

#include "AnalysisDataFormats/TopObjects/interface/TtFullLeptonicEvent.h"
#include "AnalysisDataFormats/TopObjects/interface/TtSemiLeptonicEvent.h"

typedef TtEvtBuilder< TtFullLeptonicEvent > TtFullLepEvtBuilder;
typedef TtEvtBuilder< TtSemiLeptonicEvent > TtSemiLepEvtBuilder;

DEFINE_FWK_MODULE(TtFullLepEvtBuilder);
DEFINE_FWK_MODULE(TtSemiLepEvtBuilder);
