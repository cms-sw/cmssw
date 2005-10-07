#include "TBDataFormats/EcalTBObjects/interface/EcalTBTDCInfo.h"


EcalTBTDCInfo::EcalTBTDCInfo() : size_(0), data_(MAXSAMPLES) {
}
  
void EcalTBTDCInfo::setSize(int size) {
  if (size<0) size_=0;
  else if (size>MAXSAMPLES) size_=MAXSAMPLES;
  else size_=size;
}
  
std::ostream& operator<<(std::ostream& s, const EcalTBTDCInfo& tdcInfo) {
  s << "TDC has " << tdcInfo.size() << " samples " << std::endl;
  for (int i=0; i<tdcInfo.size(); i++) 
    s << "  " << tdcInfo.sample(i) << std::endl;
  return s;
}
