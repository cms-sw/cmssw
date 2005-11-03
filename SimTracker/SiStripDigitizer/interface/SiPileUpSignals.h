#ifndef Tracker_SiPileUpSignals_h
#define Tracker_SiPileUpSignals_h

#include "SimTracker/SiStripDigitizer/interface/SiHitDigitizer.h"
#include <map>
 
class SimHit;
/**
 * Class which takes the responses from each SimHit and piles-up them.
 */
class SiPileUpSignals{

 public:

  typedef float Amplitude;
  //
  // This first one could be useless, let's see...
  //
  typedef map< int, Amplitude, less<int> >  signal_map_type;
  typedef map< int , vector < pair < const PSimHit*, Amplitude > >, less<int> >  HitToDigisMapType;
  
  virtual ~SiPileUpSignals(){}
  
 
  SiPileUpSignals(){reset();}
  virtual void add(SiHitDigitizer::hit_map_type, const PSimHit& hit);
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
