#ifndef Tracker_SiDigitalConverter_H
#define Tracker_SiDigitalConverter_H

#include "SimTracker/SiStripDigitizer/interface/SiPileUpSignals.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "CalibFormats/SiStripObjects/interface/SiStripGain.h"
/**
 * Base class for ADC simulation.
 */
class SiDigitalConverter{
   
 public:
   
  typedef std::map< int, int, std::less<int> >         DigitalMapType;
  typedef SiPileUpSignals::signal_map_type   signal_map_type;
  
  virtual ~SiDigitalConverter() { }
  virtual DigitalMapType convert(const signal_map_type &,  edm::ESHandle<SiStripGain>& ,unsigned int detid) = 0;
};

#endif
