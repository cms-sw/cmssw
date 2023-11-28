#include "SimTracker/SiStripDigitizer/interface/SiTrivialDigitalConverter.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

SiTrivialDigitalConverter::SiTrivialDigitalConverter(float in, bool PreMix)
    : ADCperElectron(1.f / in), PreMixing_(PreMix) {
  _temp.reserve(800);
  _tempRaw.reserve(800);
}

SiDigitalConverter::DigitalVecType const& SiTrivialDigitalConverter::convert(const std::vector<float>& analogSignal,
                                                                             const SiStripGain* gain,
                                                                             unsigned int detid) {
  _temp.clear();

  if (PreMixing_) {
    for (size_t i = 0; i < analogSignal.size(); i++) {
      if (analogSignal[i] <= 0)
        continue;
      // convert analog amplitude to digital - special algorithm for PreMixing.
      // Need to keep all hits, including those at very low pulse heights.
      int adc = truncate(std::sqrt(9.0f * analogSignal[i]));
      if (adc > 0)
        _temp.push_back(SiStripDigi(i, adc));
    }
  } else if (gain) {
    SiStripApvGain::Range detGainRange = gain->getRange(detid);
    for (size_t i = 0; i < analogSignal.size(); i++) {
      if (analogSignal[i] <= 0)
        continue;
      // convert analog amplitude to digital
      int adc = convert((gain->getStripGain(i, detGainRange)) * (analogSignal[i]));
      if (adc > 0)
        _temp.push_back(SiStripDigi(i, adc));
    }
  } else {
    for (size_t i = 0; i < analogSignal.size(); i++) {
      if (analogSignal[i] <= 0)
        continue;
      // convert analog amplitude to digital
      int adc = convert(analogSignal[i]);
      if (adc > 0)
        _temp.push_back(SiStripDigi(i, adc));
    }
  }
  return _temp;
}

SiDigitalConverter::DigitalRawVecType const& SiTrivialDigitalConverter::convertRaw(
    const std::vector<float>& analogSignal, const SiStripGain* gain, unsigned int detid) {
  _tempRaw.clear();

  if (gain) {
    SiStripApvGain::Range detGainRange = gain->getRange(detid);
    for (size_t i = 0; i < analogSignal.size(); i++) {
      if (analogSignal[i] <= 0) {
        _tempRaw.push_back(SiStripRawDigi(0));
        continue;
      }
      // convert analog amplitude to digital
      int adc = convertRaw((gain->getStripGain(i, detGainRange)) * (analogSignal[i]));
      _tempRaw.push_back(SiStripRawDigi(adc));
    }
  } else {
    for (size_t i = 0; i < analogSignal.size(); i++) {
      if (analogSignal[i] <= 0) {
        _tempRaw.push_back(SiStripRawDigi(0));
        continue;
      }
      // convert analog amplitude to digital
      int adc = convertRaw(analogSignal[i]);
      _tempRaw.push_back(SiStripRawDigi(adc));
    }
  }
  return _tempRaw;
}

int SiTrivialDigitalConverter::truncate(float in_adc) const {
  //Rounding the ADC number instead of truncating it
  int adc = int(in_adc + 0.5f);
  /*
    254 ADC: 254  <= raw charge < 1023
    255 ADC: raw charge >= 1023
  */
  if (PreMixing_) {
    if (adc > 2047)
      return 1023;
    if (adc > 1022)
      return 1022;
  } else {
    if (adc > 1022)
      return 255;
    if (adc > 253)
      return 254;
  }
  //Protection
  if (adc < 0)
    return 0;
  return adc;
}

int SiTrivialDigitalConverter::truncateRaw(float in_adc) const {
  //Rounding the ADC number
  int adc = int(in_adc + 0.5f);
  if (adc > 1023)
    return 1023;
  //Protection
  if (adc < 0)
    return 0;
  return adc;
}
