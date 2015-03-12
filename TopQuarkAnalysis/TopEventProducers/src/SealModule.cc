#include "FWCore/Framework/interface/MakerMacros.h"
#include "TopQuarkAnalysis/TopEventProducers/interface/TtSemiEvtSolutionMaker.h"
#include "TopQuarkAnalysis/TopEventProducers/interface/TtDilepEvtSolutionMaker.h"
#include "TopQuarkAnalysis/TopEventProducers/interface/TtHadEvtSolutionMaker.h"
#include "TopQuarkAnalysis/TopEventProducers/interface/StEvtSolutionMaker.h"
#include "TopQuarkAnalysis/TopEventProducers/interface/TopInitSubset.h"
#include "TopQuarkAnalysis/TopEventProducers/interface/TopDecaySubset.h"
#include "TopQuarkAnalysis/TopEventProducers/interface/TtGenEventReco.h"
#include "TopQuarkAnalysis/TopEventProducers/interface/StGenEventReco.h"

DEFINE_FWK_MODULE(TopInitSubset);
DEFINE_FWK_MODULE(TopDecaySubset);
DEFINE_FWK_MODULE(TtGenEventReco);
DEFINE_FWK_MODULE(StGenEventReco);
DEFINE_FWK_MODULE(TtSemiEvtSolutionMaker);
DEFINE_FWK_MODULE(TtDilepEvtSolutionMaker);
DEFINE_FWK_MODULE(TtHadEvtSolutionMaker);
DEFINE_FWK_MODULE(StEvtSolutionMaker);

#include "AnalysisDataFormats/TopObjects/interface/TtFullHadronicEvent.h"
#include "AnalysisDataFormats/TopObjects/interface/TtFullLeptonicEvent.h"
#include "AnalysisDataFormats/TopObjects/interface/TtSemiLeptonicEvent.h"

#include "TopQuarkAnalysis/TopEventProducers/interface/TtEvtBuilder.h"

typedef TtEvtBuilder<TtFullHadronicEvent> TtFullHadEvtBuilder;
typedef TtEvtBuilder<TtFullLeptonicEvent> TtFullLepEvtBuilder;
typedef TtEvtBuilder<TtSemiLeptonicEvent> TtSemiLepEvtBuilder;

DEFINE_FWK_MODULE(TtFullHadEvtBuilder);
DEFINE_FWK_MODULE(TtFullLepEvtBuilder);
DEFINE_FWK_MODULE(TtSemiLepEvtBuilder);

#include "TopQuarkAnalysis/TopEventProducers/interface/StringCutObjectEvtFilter.h"

typedef StringCutObjectEvtFilter<TtGenEvent> TtGenEvtFilter;
typedef StringCutObjectEvtFilter<TtFullHadronicEvent> TtFullHadEvtFilter;
typedef StringCutObjectEvtFilter<TtFullLeptonicEvent> TtFullLepEvtFilter;
typedef StringCutObjectEvtFilter<TtSemiLeptonicEvent> TtSemiLepEvtFilter;

DEFINE_FWK_MODULE(TtGenEvtFilter);
DEFINE_FWK_MODULE(TtFullHadEvtFilter);
DEFINE_FWK_MODULE(TtFullLepEvtFilter);
DEFINE_FWK_MODULE(TtSemiLepEvtFilter);

#include "TopQuarkAnalysis/TopEventProducers/interface/PseudoTopProducer.h"
DEFINE_FWK_MODULE(PseudoTopProducer);
