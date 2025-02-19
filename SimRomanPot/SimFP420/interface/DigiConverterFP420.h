#ifndef DigiConverterFP420_h
#define DigiConverterFP420_h

#include "SimRomanPot/SimFP420/interface/DConverterFP420.h"

class DigiConverterFP420: public DConverterFP420{
 public:

  DigiConverterFP420(float in,int);
  
  DigitalMapType convert(const signal_map_type&);
    
 private:

  int convert(float in){return truncate(in/electronperADC);}
  int truncate(float in_adc);
  
  float electronperADC;
  int theMaxADC;
  int adcBits;
  int verbos;
};
 
#endif
