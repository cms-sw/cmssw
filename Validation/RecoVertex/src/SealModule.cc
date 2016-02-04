#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "Validation/RecoVertex/interface/PrimaryVertexAnalyzer.h"
#include "Validation/RecoVertex/interface/PrimaryVertexAnalyzer4PU.h"
#include "Validation/RecoVertex/interface/TrackParameterAnalyzer.h"
#include "Validation/RecoVertex/interface/V0Validator.h"

//DEFINE_SEAL_MODULE();

DEFINE_FWK_MODULE(PrimaryVertexAnalyzer);
DEFINE_FWK_MODULE(PrimaryVertexAnalyzer4PU);
DEFINE_FWK_MODULE(TrackParameterAnalyzer);
DEFINE_FWK_MODULE(V0Validator);
