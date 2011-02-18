#ifndef ZSuppressHPS240_h
#define ZSuppressHPS240_h

#include "SimRomanPot/SimFP420/interface/DConverterHPS240.h"


class ZSuppressHPS240{
 public:
  
  typedef  DConverterHPS240::DigitalMapType DigitalMapType;
    
  virtual ~ZSuppressHPS240() {  }
  virtual DigitalMapType zeroSuppress(const DigitalMapType&, int) = 0;
};
 
#endif
