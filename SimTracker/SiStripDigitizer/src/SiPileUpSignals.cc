#include "SimTracker/SiStripDigitizer/interface/SiPileUpSignals.h"
#include "SimDataFormats/TrackingHit/interface/PSimHit.h"

void SiPileUpSignals::resetLink(){
  theMapLink.clear();
  theCounterMapLink.clear();
}

void SiPileUpSignals::add(const std::vector<double>& locAmpl,
			  const size_t& firstChannelWithSignal, const size_t& lastChannelWithSignal,
			  const PSimHit* hit,const int& counter){
  for (size_t iChannel=firstChannelWithSignal; iChannel<lastChannelWithSignal; ++iChannel) {
    theMapLink[iChannel].push_back(std::pair < const PSimHit*, Amplitude >(hit,Amplitude(locAmpl[iChannel])));
    theCounterMapLink[iChannel].push_back(std::make_pair(hit, counter));
  }
}

