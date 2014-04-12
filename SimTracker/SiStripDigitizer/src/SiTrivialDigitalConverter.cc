#include "SimTracker/SiStripDigitizer/interface/SiTrivialDigitalConverter.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

SiTrivialDigitalConverter::SiTrivialDigitalConverter(float in) :
  electronperADC(in) {
  _temp.reserve(800);
  _tempRaw.reserve(800);
}

SiDigitalConverter::DigitalVecType
SiTrivialDigitalConverter::convert(const std::vector<float>& analogSignal, edm::ESHandle<SiStripGain> & gainHandle, unsigned int detid){
  
  _temp.clear();
  
  if(gainHandle.isValid()) {
    SiStripApvGain::Range detGainRange = gainHandle->getRange(detid);
    for ( size_t i=0; i<analogSignal.size(); i++) {
      if (analogSignal[i]<=0) continue;
      // convert analog amplitude to digital
      int adc = convert( (gainHandle->getStripGain(i, detGainRange))*(analogSignal[i]) );
      if ( adc > 0) _temp.push_back(SiStripDigi(i, adc));
    }
  } else {
    for ( size_t i=0; i<analogSignal.size(); i++) {
      if (analogSignal[i]<=0) continue;
      // convert analog amplitude to digital
      int adc = convert( analogSignal[i] );
      if ( adc > 0) _temp.push_back(SiStripDigi(i, adc));
    }
  }
  return _temp;
}

SiDigitalConverter::DigitalRawVecType
SiTrivialDigitalConverter::convertRaw(const std::vector<float>& analogSignal, edm::ESHandle<SiStripGain> & gainHandle, unsigned int detid){
  
  _tempRaw.clear();

  if(gainHandle.isValid()) {
    SiStripApvGain::Range detGainRange = gainHandle->getRange(detid);
    for ( size_t i=0; i<analogSignal.size(); i++) {
      if (analogSignal[i]<=0) { _tempRaw.push_back(SiStripRawDigi(0)); continue; }
      // convert analog amplitude to digital
      int adc = convertRaw( (gainHandle->getStripGain(i, detGainRange))*(analogSignal[i]));
      _tempRaw.push_back(SiStripRawDigi(adc));
    }
  } else {
    for ( size_t i=0; i<analogSignal.size(); i++) {
      if (analogSignal[i]<=0) { _tempRaw.push_back(SiStripRawDigi(0)); continue; }
      // convert analog amplitude to digital
      int adc = convertRaw( analogSignal[i] );
      _tempRaw.push_back(SiStripRawDigi(adc));
    }
  }
  return _tempRaw;
}

int SiTrivialDigitalConverter::truncate(float in_adc) const {
  //Rounding the ADC number instead of truncating it
  int adc = int(in_adc+0.5);
  /*
    254 ADC: 254  <= raw charge < 1023
    255 ADC: raw charge >= 1023
  */
  if (adc > 1022 ) return 255;
  if (adc > 253) return 254;
  //Protection
  if (adc < 0) return 0;
  return adc;
}

int SiTrivialDigitalConverter::truncateRaw(float in_adc) const {
  //Rounding the ADC number
  int adc = int(in_adc+0.5);
  if (adc > 1023 ) return 1023;
  //Protection
  if (adc < 0) return 0;
  return adc;
}

