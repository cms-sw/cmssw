#include "TBDataFormats/HcalTBObjects/interface/HcalTBTriggerData.h"
#include "TBDataFormats/HcalTBObjects/interface/HcalTBRunData.h"
#include "TBDataFormats/HcalTBObjects/interface/HcalTBEventPosition.h"
#include "TBDataFormats/HcalTBObjects/interface/HcalTBTiming.h"
#include "FWCore/EDProduct/interface/Wrapper.h"

namespace {
  namespace {

    edm::Wrapper<HcalTBTriggerData>   theTriggerData_;
    edm::Wrapper<HcalTBRunData>       theRunData_;
    edm::Wrapper<HcalTBEventPosition> theEvtPosData_;
    edm::Wrapper<HcalTBTiming>        theTimingData_;

 }
}

