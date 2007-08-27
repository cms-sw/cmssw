#include "SimTracker/SiStripDigitizer/interface/SiPileUpSignals.h"
#include "SimDataFormats/TrackingHit/interface/PSimHit.h"

void SiPileUpSignals::resetLink(){
  theMapLink.clear();
}

void SiPileUpSignals::add(const signal_map_type &in, const PSimHit& hit){
  for (signal_map_type::const_iterator im = in.begin(); im!=in.end(); im++ ){
    theMapLink[(*im).first].push_back(std::pair < const PSimHit*, Amplitude >(&hit,Amplitude((*im).second)));
  }
}
