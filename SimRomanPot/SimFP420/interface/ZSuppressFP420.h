#ifndef ZSuppressFP420_h
#define ZSuppressFP420_h

#include "SimRomanPot/SimFP420/interface/DConverterFP420.h"

class ZSuppressFP420 {
public:
  typedef DConverterFP420::DigitalMapType DigitalMapType;

  virtual ~ZSuppressFP420() {}
  virtual DigitalMapType zeroSuppress(const DigitalMapType &, int) = 0;
};

#endif
