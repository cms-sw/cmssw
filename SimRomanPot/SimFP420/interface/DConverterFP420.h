#ifndef DConverterFP420_h
#define DConverterFP420_h

#include "SimRomanPot/SimFP420/interface/PileUpFP420.h"
class DConverterFP420{
  // ADC simulation
 public:
   
  typedef std::map<int, int, std::less<int> >         DigitalMapType;
  typedef PileUpFP420::signal_map_type   signal_map_type;
  
  virtual ~DConverterFP420() { }
  virtual DigitalMapType convert(const signal_map_type &) = 0;
};

#endif
