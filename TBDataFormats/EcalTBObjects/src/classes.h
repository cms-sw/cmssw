#include "TBDataFormats/EcalTBObjects/interface/EcalTBTDCInfo.h"
#include "FWCore/EDProduct/interface/Wrapper.h"

namespace {
  namespace {
    std::vector<EcalTBTDCSample> vTDC_;
    
    EcalTBTDCInfo TDCw_;

    edm::Wrapper<EcalTBTDCInfo> theTDCw_;

 }
}

