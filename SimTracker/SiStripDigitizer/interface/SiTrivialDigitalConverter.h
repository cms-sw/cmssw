#ifndef Tracker_SiTrivialDigitalConverter_H
#define Tracker_SiTrivialDigitalConverter_H

#include "SimTracker/SiStripDigitizer/interface/SiDigitalConverter.h"
/**
 * Concrete implementation of SiDigitalConverter.
 */
class SiTrivialDigitalConverter: public SiDigitalConverter{
 public:

  SiTrivialDigitalConverter(float in);
  
  DigitalVecType    convert(const signal_map_type&,  edm::ESHandle<SiStripGain>& ,unsigned int);    
  DigitalRawVecType convertRaw(const signal_map_type&,  edm::ESHandle<SiStripGain>& ,unsigned int);    

 private:

  int convert(float in){return truncate(in/electronperADC);}
  int truncate(float in_adc);
  
  float electronperADC;
};
 
#endif
