#include "PluginManager/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "Validation/VertexReco/interface/TrackParameterAnalyzer.h"
#include "Validation/VertexReco/interface/PrimaryVertexAnalyzer.h"

DEFINE_SEAL_MODULE();

DEFINE_ANOTHER_FWK_MODULE(PrimaryVertexAnalyzer);
DEFINE_ANOTHER_FWK_MODULE(TrackParameterAnalyzer);
