#include "SimTracker/SiStripDigitizer/interface/SiPileUpSignals.h"
#include "SimDataFormats/TrackingHit/interface/PSimHit.h"

void SiPileUpSignals::resetSignal(){
  theMap.clear();
}
void SiPileUpSignals::resetLink(){
  theMapLink.clear();
}
void SiPileUpSignals::add(SiHitDigitizer::hit_map_type in, const PSimHit& hit){
  for (SiHitDigitizer::hit_map_type::const_iterator im = in.begin(); im!=in.end(); im++ ){
    theMap[(*im).first] += Amplitude((*im).second);
    theMapLink[(*im).first].push_back(pair < const PSimHit*, Amplitude >(&hit,Amplitude((*im).second)));
  }
}
