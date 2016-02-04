#include "TBDataFormats/EcalTBObjects/interface/EcalTBEventHeader.h"
#include "TBDataFormats/EcalTBObjects/interface/EcalTBTDCRawInfo.h"
#include "TBDataFormats/EcalTBObjects/interface/EcalTBTDCRecInfo.h"
#include "TBDataFormats/EcalTBObjects/interface/EcalTBHodoscopeRawInfo.h"
#include "TBDataFormats/EcalTBObjects/interface/EcalTBHodoscopeRecInfo.h"
#include "DataFormats/Common/interface/Wrapper.h"

namespace {
  struct dictionary {
    std::vector<EcalTBTDCSample> vTDC_;
    std::vector<EcalTBHodoscopePlaneRawHits> vHplaneRawHits_;
    
    EcalTBEventHeader EHw_;
 
    EcalTBEventHeader::magnetsMeasurement aMagnetMeasurement_;
    //    EcalTBEventHeader::magnetsMeasurement_t aMagnetMeasurement_t;

    std::vector< EcalTBEventHeader::magnetsMeasurement > aVMagnetMeasurement_;
    //    std::vector< EcalTBEventHeader::magnetsMeasurement_t > aVMagnetMeasurement_t;

    EcalTBTDCRawInfo TDCw_;
    EcalTBTDCRecInfo RecTDCw_;

    EcalTBHodoscopeRawInfo Hodow_;
    EcalTBHodoscopeRecInfo RecHodow_;

    edm::Wrapper<EcalTBEventHeader> theEHw_;

    edm::Wrapper<EcalTBTDCRawInfo> theTDCw_;
    edm::Wrapper<EcalTBTDCRecInfo> theRecTDCw_;

    edm::Wrapper<EcalTBHodoscopeRawInfo> theHodow_;
    edm::Wrapper<EcalTBHodoscopeRecInfo> theRecHodow_;

 };
}

