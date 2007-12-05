#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "Validation/RecoParticleFlow/interface/PFTester.h"
#include "Validation/RecoParticleFlow/interface/PFBenchmarkAnalyzer.h"

DEFINE_SEAL_MODULE();

DEFINE_ANOTHER_FWK_MODULE (PFTester) ;
DEFINE_ANOTHER_FWK_MODULE (PFBenchmarkAnalyzer) ;
