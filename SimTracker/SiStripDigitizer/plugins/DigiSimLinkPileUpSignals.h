#ifndef SimTracker_SiStripDigitizer_DigiSimLinkPileUpSignals_h
#define SimTracker_SiStripDigitizer_DigiSimLinkPileUpSignals_h

#include <map>
#include <vector>
#include "SimDataFormats/TrackingHit/interface/PSimHit.h"

class SimHit;
/**
 * Class which takes the responses from each SimHit and piles-up them, within a given module.
 * More precisely, it keeps for each strip the link to each individual measurement.
 */
class DigiSimLinkPileUpSignals{ 
 public:
  // type used to describe the amplitude on a strip
  typedef float Amplitude;
  // associates to each strip a vector of pairs {simhit,amplitude}. 
  // That allows later to comput the fraction of the contribution of each simhit to the ADC value
  typedef std::map< int , std::vector < std::pair < const PSimHit*, Amplitude > >, std::less<int> >  HitToDigisMapType;
  // associates to each strip a vector of pairs {simhit,index_in_the_allhit_collection}
  // That allows to build the links properly
  typedef std::map< int , std::vector < std::pair < const PSimHit*, int > >, std::less<int> >  HitCounterToDigisMapType;
  
  DigiSimLinkPileUpSignals() { reset(); }

  virtual ~DigiSimLinkPileUpSignals() { }
  
  virtual void add(const std::vector<float>& locAmpl,
		   const size_t& firstChannelWithSignal, const size_t& lastChannelWithSignal,
		   const PSimHit* hit,const int& counter);
		   
  void reset(){ resetLink(); }
  
  const HitToDigisMapType& dumpLink() const { return theMapLink; }
  
  const HitCounterToDigisMapType& dumpCounterLink() const { return theCounterMapLink; }
  
 private:
  void resetLink();
  HitToDigisMapType theMapLink;
  HitCounterToDigisMapType theCounterMapLink;
};
#endif
