#include "TBDataFormats/EcalTBObjects/interface/EcalTBTDCRecInfo.h"


std::ostream& operator<<(std::ostream& s, const EcalTBTDCRecInfo& tdcInfo) {
  s << "TDC offset is " << tdcInfo.offset() ;
  return s;
}
