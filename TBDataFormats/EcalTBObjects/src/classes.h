#include "TBDataFormats/EcalTBObjects/interface/EcalTBTDCRawInfo.h"
#include "TBDataFormats/EcalTBObjects/interface/EcalTBHodoscopeRawInfo.h"
#include "DataFormats/Common/interface/Wrapper.h"

namespace {
  namespace {
    std::vector<EcalTBTDCSample> vTDC_;
    std::vector<EcalTBHodoscopePlaneRawHits> vHplaneRawHits_;
    
    EcalTBTDCRawInfo TDCw_;
    EcalTBHodoscopeRawInfo Hodow_;

    edm::Wrapper<EcalTBTDCRawInfo> theTDCw_;
    edm::Wrapper<EcalTBHodoscopeRawInfo> theHodow_;

 }
}

