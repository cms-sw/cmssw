#ifndef Tracker_SiTrivialDigitalConverter_H
#define Tracker_SiTrivialDigitalConverter_H

#include "SimTracker/SiStripDigitizer/interface/SiDigitalConverter.h"
/**
 * Concrete implementation of SiDigitalConverter.
 */
class SiTrivialDigitalConverter : public SiDigitalConverter {
public:
  SiTrivialDigitalConverter(float in, bool PreMix);

  DigitalVecType convert(const std::vector<float>&, const SiStripGain*, unsigned int detid) override;
  DigitalRawVecType convertRaw(const std::vector<float>&, const SiStripGain*, unsigned int detid) override;

private:
  int convert(float in) { return truncate(in * ADCperElectron); }
  int convertRaw(float in) { return truncateRaw(in * ADCperElectron); }
  int truncate(float in_adc) const;
  int truncateRaw(float in_adc) const;

  const float ADCperElectron;
  SiDigitalConverter::DigitalVecType _temp;
  SiDigitalConverter::DigitalRawVecType _tempRaw;
  bool PreMixing_;
};

#endif
