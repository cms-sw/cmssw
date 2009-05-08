#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "Validation/RecoVertex/interface/PrimaryVertexAnalyzer.h"
#include "Validation/RecoVertex/interface/TrackParameterAnalyzer.h"
#include "Validation/RecoVertex/interface/V0Validator.h"

DEFINE_SEAL_MODULE();

DEFINE_ANOTHER_FWK_MODULE(PrimaryVertexAnalyzer);
DEFINE_ANOTHER_FWK_MODULE(TrackParameterAnalyzer);
DEFINE_ANOTHER_FWK_MODULE(V0Validator);
