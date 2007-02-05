
#include "PluginManager/ModuleDef.h"

#include "FWCore/Framework/interface/MakerMacros.h"

#include "RecoTracker/RoadMapESSource/test/RoadPainter.h"

using cms::RoadPainter;

DEFINE_SEAL_MODULE();
DEFINE_ANOTHER_FWK_MODULE(RoadPainter);

