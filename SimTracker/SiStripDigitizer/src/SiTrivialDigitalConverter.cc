#include "SimTracker/SiStripDigitizer/interface/SiTrivialDigitalConverter.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

SiTrivialDigitalConverter::SiTrivialDigitalConverter(float in){
  electronperADC = in;
}

SiDigitalConverter::DigitalVecType
SiTrivialDigitalConverter::convert(const signal_map_type& analogSignal, edm::ESHandle<SiStripGain> & gainHandle, unsigned int detid){

  SiDigitalConverter::DigitalVecType _temp;
  _temp.reserve(analogSignal.size());

  SiStripApvGain::Range detGainRange; 
  if(gainHandle.isValid()) detGainRange = gainHandle->getRange(detid);


  for ( signal_map_type::const_iterator i=analogSignal.begin(); 
	i!=analogSignal.end(); i++) {
    float gainFactor  = (gainHandle.isValid()) ? gainHandle->getStripGain((*i).first, detGainRange) : 1;

    // convert analog amplitude to digital
    int adc = convert( gainFactor*((*i).second));
     
    if ( adc > 0) _temp.push_back(SiStripDigi((*i).first, adc));
  }
  return _temp;
}
SiDigitalConverter::DigitalRawVecType
SiTrivialDigitalConverter::convertRaw(const signal_map_type& analogSignal, edm::ESHandle<SiStripGain> & gainHandle, unsigned int detid){

  SiDigitalConverter::DigitalRawVecType _temp;
  _temp.reserve(analogSignal.size());

  SiStripApvGain::Range detGainRange; 
  if(gainHandle.isValid()) detGainRange = gainHandle->getRange(detid);


  for ( signal_map_type::const_iterator i=analogSignal.begin(); 
	i!=analogSignal.end(); i++) {
    float gainFactor  = (gainHandle.isValid()) ? gainHandle->getStripGain((*i).first, detGainRange) : 1;

    // convert analog amplitude to digital
    int adc = convert( gainFactor*((*i).second));
     
    _temp.push_back(SiStripRawDigi(adc));
  }
  return _temp;
}


int SiTrivialDigitalConverter::truncate(float in_adc) {

  //Rounding teh ADC number instaed of truncating it
  int adc = int(in_adc+0.5);
  /*
    254 ADC: 254<=raw charge < 511
    255 ADC: 512< raw charge < 1023
  */
  if (adc > 253 && adc < 512) adc = 254;
  if (adc > 511 ) adc = 255;
  //Protection
  if (adc < 0) adc = 0;
  return adc;
}
