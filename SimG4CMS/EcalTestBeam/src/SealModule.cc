#include "PluginManager/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_SEAL_MODULE () ;
 
#include "SimG4CMS/EcalTestBeam/interface/EcalTBMCInfoProducer.h"
DEFINE_ANOTHER_FWK_MODULE (EcalTBMCInfoProducer) ;

#include "SimG4CMS/EcalTestBeam/interface/FakeTBHodoscopeRawInfoProducer.h"
DEFINE_ANOTHER_FWK_MODULE (FakeTBHodoscopeRawInfoProducer) ;
