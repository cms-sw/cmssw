#ifndef Tracker_SiDigitalConverter_H
#define Tracker_SiDigitalConverter_H

#include "SiPileUpSignals.h"
#include "CalibFormats/SiStripObjects/interface/SiStripGain.h"
#include "DataFormats/SiStripDigi/interface/SiStripDigi.h"
#include "DataFormats/SiStripDigi/interface/SiStripRawDigi.h"
/**
 * Base class for ADC simulation.
 */
class SiDigitalConverter {
public:
  typedef std::vector<SiStripDigi> DigitalVecType;
  typedef std::vector<SiStripRawDigi> DigitalRawVecType;

  virtual ~SiDigitalConverter() {}
  virtual DigitalVecType const& convert(const std::vector<float>&, const SiStripGain*, unsigned int detid) = 0;
  virtual DigitalRawVecType const& convertRaw(const std::vector<float>&, const SiStripGain*, unsigned int detid) = 0;
};

#endif
