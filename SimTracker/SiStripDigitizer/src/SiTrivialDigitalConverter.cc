#include "SimTracker/SiStripDigitizer/interface/SiTrivialDigitalConverter.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

SiTrivialDigitalConverter::SiTrivialDigitalConverter(float in,int fs){
  electronperADC = in;
  theMaxADC = fs; 

  /*
  // N.B. Default value of adcBits should really be 10, but is left equal
  // to 12 for reasons of backwards compatibility.
  const int defaultBits = 10;
  const int largestBits = 30;
  
  // static SimpleConfigurable<int> 
  //    adcBits(defaultBits, "SiTrivialDigitalConverter:rawDataAdcBits");
  adcBits=defaultBits;

  if (adcBits > largestBits || adcBits < 1) adcBits = largestBits;
  
  theMaxADC = ~(~0 << adcBits);
  */

}

SiDigitalConverter::DigitalMapType
SiTrivialDigitalConverter::convert(const signal_map_type& analogSignal, edm::ESHandle<SiStripGain> & gainHandle, unsigned int detid){

  SiDigitalConverter::DigitalMapType _temp;

  SiStripApvGain::Range detGainRange; 
  if(gainHandle.isValid()) detGainRange = gainHandle->getRange(detid);


  for ( signal_map_type::const_iterator i=analogSignal.begin(); 
	i!=analogSignal.end(); i++) {
    float gainFactor  = (gainHandle.isValid()) ? gainHandle->getStripGain((*i).first, detGainRange) : 1;
    edm::LogInfo("SiTrivialDigitalConverter : Gain factor =") << gainFactor <<std::endl;


    // convert analog amplitude to digital
    int adc = convert( gainFactor*((*i).second));
     
    if ( adc > 0) _temp.insert( _temp.end(),
				DigitalMapType::value_type((*i).first, adc));
  }
  return _temp;
}


int SiTrivialDigitalConverter::truncate(float in_adc) {
 
  int adc = int(in_adc);
  if (adc > theMaxADC) adc = theMaxADC;
  
  return adc;
}
