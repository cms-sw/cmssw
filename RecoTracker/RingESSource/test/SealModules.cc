
#include "PluginManager/ModuleDef.h"

#include "FWCore/Framework/interface/MakerMacros.h"

#include "RecoTracker/RingESSource/test/RingPainter.h"

using cms::RingPainter;

DEFINE_SEAL_MODULE();
DEFINE_ANOTHER_FWK_MODULE(RingPainter);

