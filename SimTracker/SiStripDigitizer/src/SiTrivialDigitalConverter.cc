#include "SimTracker/SiStripDigitizer/interface/SiTrivialDigitalConverter.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

SiTrivialDigitalConverter::SiTrivialDigitalConverter(float in, bool PreMix)
    : ADCperElectron(1.f / in), PreMixing_(PreMix) {
  _temp.reserve(800);
  _tempRaw.reserve(800);
}

namespace {

  inline int truncate(float in_adc) {
    //Rounding the ADC number instead of truncating it
    int radc = int(in_adc + 0.5f);
    /*
    254 ADC: 254  <= raw charge < 1023
    255 ADC: raw charge >= 1023
  */
    int adc = std::clamp(radc, 0, 254);
    if (radc > 1022)
      adc = 255;
    return adc;
  }

  inline int truncateForPremix(float in_adc) {
    //Rounding the ADC number instead of truncating it (in_adc is >=0)
    int radc = int(in_adc + 0.5f);
    /*
    1022 ADC: 1022  <= raw charge < 2047
    1023 ADC: raw charge > 2047
  */
    int adc = std::min(radc, 1022);
    if (radc > 2047)
      adc = 1023;
    return adc;
  }

  inline int truncateRaw(float in_adc) {
    //Rounding the ADC number
    int adc = int(in_adc + 0.5f);
    return std::clamp(adc, 0, 1023);
  }

}  // namespace

SiDigitalConverter::DigitalVecType const& SiTrivialDigitalConverter::convert(const std::vector<float>& analogSignal,
                                                                             const SiStripGain* gain,
                                                                             unsigned int detid) {
  _temp.clear();

  auto convert = [&](float in) { return truncate(in * ADCperElectron); };

  if (PreMixing_) {
    for (size_t i = 0; i < analogSignal.size(); i++) {
      if (analogSignal[i] <= 0)
        continue;
      // convert analog amplitude to digital - special algorithm for PreMixing.
      // Need to keep all hits, including those at very low pulse heights.
      int adc = truncateForPremix(std::sqrt(9.0f * analogSignal[i]));
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

  auto convertRaw = [&](float in) { return truncateRaw(in * ADCperElectron); };

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
