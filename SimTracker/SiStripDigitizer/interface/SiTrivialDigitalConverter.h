#ifndef Tracker_SiTrivialDigitalConverter_H
#define Tracker_SiTrivialDigitalConverter_H

#include "SimTracker/SiStripDigitizer/interface/SiDigitalConverter.h"
/**
 * Concrete implementation of SiDigitalConverter.
 */
class SiTrivialDigitalConverter: public SiDigitalConverter{
 public:

  SiTrivialDigitalConverter(float in);
  
  DigitalVecType    convert(const std::vector<double>&,  edm::ESHandle<SiStripGain>& ,unsigned int detid);
  DigitalRawVecType convertRaw(const std::vector<double>&,  edm::ESHandle<SiStripGain>& ,unsigned int detid);  

 private:

  int convert(float in){return truncate(in/electronperADC);}
  int truncate(float in_adc);
  
  float electronperADC;
};
 
#endif
