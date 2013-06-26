#include "TBDataFormats/EcalTBObjects/interface/EcalTBTDCRawInfo.h"


std::ostream& operator<<(std::ostream& s, const EcalTBTDCRawInfo& tdcInfo) {
  s << "TDC has " << tdcInfo.size() << " samples " << std::endl;
  for (unsigned int i=0; i<tdcInfo.size(); i++) 
    s << "  " << tdcInfo.sample(i) << std::endl;
  return s;
}
