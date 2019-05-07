#include "FWCore/PluginManager/interface/ModuleDef.h"

#include "FWCore/Framework/interface/MakerMacros.h"

#include <Validation/GlobalHits/interface/GlobalHitsProducer.h>
DEFINE_FWK_MODULE(GlobalHitsProducer);

#include <Validation/GlobalHits/interface/GlobalHitsAnalyzer.h>
DEFINE_FWK_MODULE(GlobalHitsAnalyzer);

#include <Validation/GlobalHits/interface/GlobalHitsHistogrammer.h>
DEFINE_FWK_MODULE(GlobalHitsHistogrammer);

#include <Validation/GlobalHits/interface/GlobalHitsProdHist.h>
DEFINE_FWK_MODULE(GlobalHitsProdHist);

#include <Validation/GlobalHits/interface/GlobalHitsProdHistStripper.h>
DEFINE_FWK_MODULE(GlobalHitsProdHistStripper);

#include <Validation/GlobalHits/interface/GlobalHitsTester.h>
DEFINE_FWK_MODULE(GlobalHitsTester);
