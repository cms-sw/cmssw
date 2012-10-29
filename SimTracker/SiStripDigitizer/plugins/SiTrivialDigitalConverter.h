#ifndef Tracker_SiTrivialDigitalConverter_H
#define Tracker_SiTrivialDigitalConverter_H

#include "SiDigitalConverter.h"
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
  int convertRaw(float in){return truncateRaw(in/electronperADC);}
  int truncate(float in_adc) const;
  int truncateRaw(float in_adc) const;
  
  const float electronperADC;
  SiDigitalConverter::DigitalVecType _temp;
  SiDigitalConverter::DigitalRawVecType _tempRaw;

};
 
#endif
