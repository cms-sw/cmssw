#include "PluginManager/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
 
#include "SimG4CMS/EcalTestBeam/interface/EcalTBMCInfoProducer.h"

DEFINE_SEAL_MODULE () ;
DEFINE_ANOTHER_FWK_MODULE (EcalTBMCInfoProducer) ;
