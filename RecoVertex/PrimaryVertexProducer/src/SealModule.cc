#include "PluginManager/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "RecoVertex/PrimaryVertexProducer/interface/PrimaryVertexProducer.h"
#include "RecoVertex/PrimaryVertexProducer/interface/PrimaryVertexAnalyzer.h"
#include "RecoVertex/PrimaryVertexProducer/interface/TrackParameterAnalyzer.h"

DEFINE_SEAL_MODULE();

DEFINE_ANOTHER_FWK_MODULE(PrimaryVertexProducer);
DEFINE_ANOTHER_FWK_MODULE(PrimaryVertexAnalyzer);
DEFINE_ANOTHER_FWK_MODULE(TrackParameterAnalyzer);
