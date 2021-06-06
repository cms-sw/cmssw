#include "FWCore/Framework/interface/MakerMacros.h"

#include "EcalTPCondAnalyzer.h"
#include "EcalTPInputAnalyzer.h"
#include "EcalTrigPrimAnalyzer.h"
#include "EcalTrigPrimESProducer.h"
#include "EcalTrigPrimSpikeESProducer.h"

DEFINE_FWK_MODULE(EcalTPInputAnalyzer);
DEFINE_FWK_MODULE(EcalTrigPrimAnalyzer);
DEFINE_FWK_MODULE(EcalTPCondAnalyzer);
DEFINE_FWK_EVENTSETUP_MODULE(EcalTrigPrimESProducer);
DEFINE_FWK_EVENTSETUP_MODULE(EcalTrigPrimSpikeESProducer);
