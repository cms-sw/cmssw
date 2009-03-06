#ifndef PileUpFP420_h
#define PileUpFP420_h

#include "SimRomanPot/SimFP420/interface/HitDigitizerFP420.h"
#include <map>

class  SimHit;

// Class which takes the responses from each SimHit and piles-up them.
class PileUpFP420{

 public:

  typedef float Amplitude;
  typedef map< int, Amplitude, less<int> >  signal_map_type;
  typedef map< int , vector < pair < const PSimHit*, Amplitude > >, less<int> >  HitToDigisMapType;

  virtual ~PileUpFP420(){}
  
  PileUpFP420(){reset();}
  virtual void add(HitDigitizerFP420::hit_map_type, const PSimHit& hit);
  void reset(){resetLink();resetSignal();}
  signal_map_type dumpSignal() {return theMap;}
  HitToDigisMapType dumpLink() {return theMapLink;}
 private:
  void resetLink();
  void resetSignal();
  HitToDigisMapType theMapLink;
  signal_map_type theMap;

};
#endif
