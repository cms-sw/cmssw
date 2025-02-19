#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "Validation/RecoParticleFlow/plugins/PFTester.h"
#include "Validation/RecoParticleFlow/plugins/GenericBenchmarkAnalyzer.h"
#include "Validation/RecoParticleFlow/plugins/PFMETFilter.h"
#include "Validation/RecoParticleFlow/plugins/PFJetFilter.h"



DEFINE_FWK_MODULE (PFTester) ;
DEFINE_FWK_MODULE (GenericBenchmarkAnalyzer) ;
DEFINE_FWK_MODULE (PFJetFilter) ;
DEFINE_FWK_MODULE (PFMETFilter) ;
