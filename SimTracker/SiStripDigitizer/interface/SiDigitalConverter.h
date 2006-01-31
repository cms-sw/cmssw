#ifndef Tracker_SiDigitalConverter_H
#define Tracker_SiDigitalConverter_H

#include "SimTracker/SiStripDigitizer/interface/SiPileUpSignals.h"
/**
 * Base class for ADC simulation.
 */
class SiDigitalConverter{
   
 public:
   
  typedef map< int, int, less<int> >         DigitalMapType;
  typedef SiPileUpSignals::signal_map_type   signal_map_type;
  
  virtual ~SiDigitalConverter() { }
  virtual DigitalMapType convert(const signal_map_type &) = 0;
};

#endif
