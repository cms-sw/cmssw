#include "PluginManager/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "Validation/RecoVertex/interface/PrimaryVertexAnalyzer.h"
#include "Validation/RecoVertex/interface/TrackParameterAnalyzer.h"

DEFINE_SEAL_MODULE();

DEFINE_ANOTHER_FWK_MODULE(PrimaryVertexAnalyzer);
DEFINE_ANOTHER_FWK_MODULE(TrackParameterAnalyzer);
