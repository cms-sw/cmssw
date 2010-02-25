
//#include "FWCore/PluginManager/interface/ModuleDef.h"

#include "FWCore/Framework/interface/MakerMacros.h"

#include "EcalTrigPrimProducer.h"
#include "EcalTrigPrimAnalyzer.h"
#include "EcalTrigPrimAnalyzerMIPs.h"
#include "EcalTPInputAnalyzer.h"
#include "EcalTPCondAnalyzer.h"
#include "EcalTrigPrimESProducer.h"



DEFINE_FWK_MODULE(EcalTrigPrimProducer);
DEFINE_FWK_MODULE(EcalTPInputAnalyzer);
DEFINE_FWK_MODULE(EcalTrigPrimAnalyzer);
DEFINE_FWK_MODULE(EcalTrigPrimAnalyzerMIPs);
DEFINE_FWK_MODULE(EcalTPCondAnalyzer);
DEFINE_FWK_EVENTSETUP_MODULE(EcalTrigPrimESProducer);
