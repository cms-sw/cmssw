#ifndef DConverterHPS240_h
#define DConverterHPS240_h

#include "SimRomanPot/SimFP420/interface/PileUpFP420.h"
class DConverterHPS240{
  // ADC simulation
 public:
   
  typedef std::map< int, int, std::less<int> >         DigitalMapType;
  typedef PileUpFP420::signal_map_type   signal_map_type;
  
  virtual ~DConverterHPS240() { }
  virtual DigitalMapType convert(const signal_map_type &) = 0;
};

#endif
