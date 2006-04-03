#include "PluginManager/ModuleDef.h"

#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_SEAL_MODULE();

#include <Validation/EcalDigis/interface/EcalDigisValidation.h>
DEFINE_ANOTHER_FWK_MODULE(EcalDigisValidation)
