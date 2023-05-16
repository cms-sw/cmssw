#ifndef Tracker_SiTrivialDigitalConverter_H
#define Tracker_SiTrivialDigitalConverter_H

#include "SimTracker/SiStripDigitizer/interface/SiDigitalConverter.h"
/**
 * Concrete implementation of SiDigitalConverter.
 */
class SiTrivialDigitalConverter final : public SiDigitalConverter {
public:
  SiTrivialDigitalConverter(float in, bool PreMix);

  DigitalVecType const& convert(const std::vector<float>&, const SiStripGain*, unsigned int detid) override;
  DigitalRawVecType const& convertRaw(const std::vector<float>&, const SiStripGain*, unsigned int detid) override;

private:
  const float ADCperElectron;
  SiDigitalConverter::DigitalVecType _temp;
  SiDigitalConverter::DigitalRawVecType _tempRaw;
  bool PreMixing_;
};

#endif
