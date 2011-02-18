#ifndef DigiConverterHPS240_h
#define DigiConverterHPS240_h

#include "SimRomanPot/SimFP420/interface/DConverterHPS240.h"

class DigiConverterHPS240: public DConverterHPS240{
 public:

  DigiConverterHPS240(float in,int);
  
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
