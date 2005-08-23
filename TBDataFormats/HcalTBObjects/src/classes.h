#include "TBDataFormats/HcalTBObjects/interface/HcalTBTriggerData.h"
#include "TBDataFormats/HcalTBObjects/interface/HcalTBRunData.h"
#include "TBDataFormats/HcalTBObjects/interface/HcalTBEventPosition.h"
#include "TBDataFormats/HcalTBObjects/interface/HcalTBTiming.h"
#include "FWCore/EDProduct/interface/Wrapper.h"

namespace {
  namespace {

    edm::Wrapper<hcaltb::HcalTBTriggerData>   theTriggerData_;
    edm::Wrapper<hcaltb::HcalTBRunData>       theRunData_;
    edm::Wrapper<hcaltb::HcalTBEventPosition> theEvtPosData_;
    edm::Wrapper<hcaltb::HcalTBTiming>        theTimingData_;

 }
}

