#include "TBDataFormats/EcalTBObjects/interface/EcalTBTDCSample.h"


EcalTBTDCSample::EcalTBTDCSample(int tdcChan, int tdcVal) {
  theSample=(tdcVal&0xFFFFFF) | ((tdcChan&0xFF)<<24);
}

std::ostream& operator<<(std::ostream& s, const EcalTBTDCSample& samp) {
  s << "TDC Channel=" << samp.tdcChannel() << ", Value=" << samp.tdcValue();
  return s;
}
