#include "TBDataFormats/HcalTBObjects/interface/HcalTBTriggerData.h"
#include "TBDataFormats/HcalTBObjects/interface/HcalTBRunData.h"
#include "TBDataFormats/HcalTBObjects/interface/HcalTBEventPosition.h"
#include "TBDataFormats/HcalTBObjects/interface/HcalTBTiming.h"
#include "TBDataFormats/HcalTBObjects/interface/HcalTBBeamCounters.h"
#include "TBDataFormats/HcalTBObjects/interface/HcalTBParticleId.h"
#include "DataFormats/Common/interface/Wrapper.h"

namespace {
  struct dictionary {

    edm::Wrapper<HcalTBTriggerData>   theTriggerData_;
    edm::Wrapper<HcalTBRunData>       theRunData_;
    edm::Wrapper<HcalTBEventPosition> theEvtPosData_;
    edm::Wrapper<HcalTBTiming>        theTimingData_;
    edm::Wrapper<HcalTBBeamCounters>  theBeamCountersData_;
    edm::Wrapper<HcalTBParticleId>    theParticleId_;

 };
}

