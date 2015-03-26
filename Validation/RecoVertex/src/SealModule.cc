#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "Validation/RecoVertex/interface/PrimaryVertexAnalyzer4PUSlimmed.h"
#include "Validation/RecoVertex/interface/TrackParameterAnalyzer.h"
#include "Validation/RecoVertex/interface/V0Validator.h"

//DEFINE_SEAL_MODULE();

DEFINE_FWK_MODULE(PrimaryVertexAnalyzer4PUSlimmed);
DEFINE_FWK_MODULE(TrackParameterAnalyzer);
DEFINE_FWK_MODULE(V0Validator);
