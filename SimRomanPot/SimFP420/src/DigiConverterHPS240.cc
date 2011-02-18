///////////////////////////////////////////////////////////////////////////////
// File: DigiConverterHPS240.cc
// Date: 12.2006
// Description: DigiConverterHPS240 for HPS240
// Modifications: 
///////////////////////////////////////////////////////////////////////////////
#include "SimRomanPot/SimFP420/interface/DigiConverterHPS240.h"

DigiConverterHPS240::DigiConverterHPS240(float in,int verbosity){

  electronperADC = in;
  verbos = verbosity;
  
  const int defaultBits = 10;
  const int largestBits = 30;
  
  // example is in SiStrips:
  // static SimpleConfigurable<int> 
  //    adcBits(defaultBits, "DigiConverterHPS240:rawDataAdcBits");
  adcBits=defaultBits;
  
  if (adcBits > largestBits || adcBits < 1) adcBits = largestBits;
  
  theMaxADC = ~(~0 << adcBits);
  //      cout << "theMaxADC= "<< theMaxADC  << endl; // = 1023
  if(verbos>0) {
    std::cout << " ***DigiConverterHPS240: constructor" << std::endl;
    std::cout << "with known electronperADC =  " << electronperADC << "the adcBits =  " << adcBits << "  theMaxADC=  " << theMaxADC << "for known defaultBits=  " << defaultBits << " largestBits=  " << largestBits << std::endl;
  }
}

DConverterHPS240::DigitalMapType
DigiConverterHPS240::convert(const signal_map_type& analogSignal){
  
  DConverterHPS240::DigitalMapType _temp;
  
  for ( signal_map_type::const_iterator i=analogSignal.begin(); i!=analogSignal.end(); i++) {
    
    // convert analog amplitude to digital, means integer number simulating ADC digitization!
    //with truncation check
    int adc = convert((*i).second);
    
    if(verbos>0) {
      std::cout << " ***DigiConverterHPS240: convert: after truncation " << std::endl;
      std::cout << "adc =  " << adc << " (*i).first =  " << (*i).first << std::endl;
    }
    if ( adc > 0) _temp.insert( _temp.end(),
				DigitalMapType::value_type((*i).first, adc));
  }
  
  return _temp;
  
}


int DigiConverterHPS240::truncate(float in_adc) {
  
  int adc = int(in_adc);
  if(verbos>0) {
    std::cout << " ***DigiConverterHPS240: truncate" << std::endl;
    std::cout << "if adc =  " << adc << "bigger theMaxADC =  " << theMaxADC << " adc=theMaxADC !!!"  << std::endl;
  }
  if (adc > theMaxADC) adc = theMaxADC;
  
  return adc;
}
