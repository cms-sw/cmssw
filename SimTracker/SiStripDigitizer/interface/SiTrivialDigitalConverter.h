#ifndef Tracker_SiTrivialDigitalConverter_H
#define Tracker_SiTrivialDigitalConverter_H

#include "SimTracker/SiStripDigitizer/interface/SiDigitalConverter.h"
/**
 * Concrete implementation of SiDigitalConverter.
 */
class SiTrivialDigitalConverter: public SiDigitalConverter{
 public:

  SiTrivialDigitalConverter(float in,int fs);
  
  DigitalMapType convert(const signal_map_type&,  edm::ESHandle<SiStripGain>& ,unsigned int detid);
    
 private:

  int convert(float in){return truncate(in/electronperADC);}
  int truncate(float in_adc);
  
  float electronperADC;
  int theMaxADC;
  int adcBits;
};
 
#endif
