
//#include "FWCore/PluginManager/interface/ModuleDef.h"

#include "FWCore/Framework/interface/MakerMacros.h"

#include "EcalTrigPrimProducer.h"
#include "EcalTrigPrimAnalyzer.h"
#include "EcalTrigPrimAnalyzerMIPs.h"
#include "EcalTPInputAnalyzer.h"
#include "EcalTrigPrimESProducer.h"


DEFINE_SEAL_MODULE();
DEFINE_ANOTHER_FWK_MODULE(EcalTrigPrimProducer);
DEFINE_ANOTHER_FWK_MODULE(EcalTPInputAnalyzer);
DEFINE_ANOTHER_FWK_MODULE(EcalTrigPrimAnalyzer);
DEFINE_ANOTHER_FWK_MODULE(EcalTrigPrimAnalyzerMIPs);
DEFINE_ANOTHER_FWK_EVENTSETUP_MODULE(EcalTrigPrimESProducer);
