#ifndef Tracker_SiPileUpSignals_h
#define Tracker_SiPileUpSignals_h

#include <map>
#include <vector>
#include "SimDataFormats/TrackingHit/interface/PSimHit.h"

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
  typedef std::map< int, Amplitude, std::less<int> >  signal_map_type;
  typedef std::map< int , std::vector < std::pair < const PSimHit*, Amplitude > >, std::less<int> >  HitToDigisMapType;
  typedef std::map< int , std::vector < std::pair < const PSimHit*, int > >, std::less<int> >  HitCounterToDigisMapType;
  
  virtual ~SiPileUpSignals(){}
  
  
  SiPileUpSignals(){reset();}
  virtual void add(const std::vector<double>& locAmpl,
		   const unsigned int& firstChannelWithSignal, const unsigned int& lastChannelWithSignal,
		   const PSimHit& hit,const int& counter);
  void reset(){resetLink(); }
  const HitToDigisMapType& dumpLink() const {return theMapLink;}
  const HitCounterToDigisMapType& dumpCounterLink() const {return theCounterMapLink;}
  
private:
  void resetLink();
  HitToDigisMapType theMapLink;
  HitCounterToDigisMapType theCounterMapLink;
};
#endif
