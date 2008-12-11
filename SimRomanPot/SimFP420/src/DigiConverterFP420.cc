///////////////////////////////////////////////////////////////////////////////
// File: DigiConverterFP420.cc
// Date: 12.2006
// Description: DigiConverterFP420 for FP420
// Modifications: 
///////////////////////////////////////////////////////////////////////////////
#include "SimRomanPot/SimFP420/interface/DigiConverterFP420.h"
//#define mydigidebug8

DigiConverterFP420::DigiConverterFP420(float in){

  electronperADC = in;
  
  const int defaultBits = 10;
  const int largestBits = 30;
  
  // example is in SiStrips:
  // static SimpleConfigurable<int> 
  //    adcBits(defaultBits, "DigiConverterFP420:rawDataAdcBits");
  adcBits=defaultBits;
  
  if (adcBits > largestBits || adcBits < 1) adcBits = largestBits;
  
  theMaxADC = ~(~0 << adcBits);
  //      cout << "theMaxADC= "<< theMaxADC  << endl; // = 1023
#ifdef mydigidebug8
  std::cout << " ***DigiConverterFP420: constructor" << std::endl;
  std::cout << "with known electronperADC =  " << electronperADC << "the adcBits =  " << adcBits << "  theMaxADC=  " << theMaxADC << "for known defaultBits=  " << defaultBits << " largestBits=  " << largestBits << std::endl;
#endif
}

DConverterFP420::DigitalMapType
DigiConverterFP420::convert(const signal_map_type& analogSignal){
  
  DConverterFP420::DigitalMapType _temp;
  
  for ( signal_map_type::const_iterator i=analogSignal.begin(); i!=analogSignal.end(); i++) {
    
    // convert analog amplitude to digital, means integer number simulating ADC digitization!
    //with truncation check
    int adc = convert((*i).second);
    
#ifdef mydigidebug8
    std::cout << " ***DigiConverterFP420: convert: after truncation " << std::endl;
    std::cout << "adc =  " << adc << " (*i).first =  " << (*i).first << std::endl;
#endif
    if ( adc > 0) _temp.insert( _temp.end(),
				DigitalMapType::value_type((*i).first, adc));
  }
  
  return _temp;
  
}


int DigiConverterFP420::truncate(float in_adc) {
  
  int adc = int(in_adc);
#ifdef mydigidebug8
  std::cout << " ***DigiConverterFP420: truncate" << std::endl;
  std::cout << "if adc =  " << adc << "bigger theMaxADC =  " << theMaxADC << " adc=theMaxADC !!!"  << std::endl;
#endif
  if (adc > theMaxADC) adc = theMaxADC;
  
  return adc;
}
