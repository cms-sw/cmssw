#ifndef Tracker_SiZeroSuppress_H
#define Tracker_SiZeroSuppress_H

#include "SimTracker/SiStripDigitizer/interface/SiDigitalConverter.h"
/**
 * Base class for Zero Suppression in Silicon u-strips.
 */
class SiZeroSuppress{
 public:
  
  typedef  SiDigitalConverter::DigitalMapType DigitalMapType;
    
  virtual DigitalMapType zeroSuppress(const DigitalMapType&) = 0;
};
 
#endif
