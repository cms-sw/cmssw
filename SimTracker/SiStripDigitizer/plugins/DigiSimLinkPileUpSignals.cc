#include "DigiSimLinkPileUpSignals.h"
#include "SimDataFormats/TrackingHit/interface/PSimHit.h"

void DigiSimLinkPileUpSignals::resetLink(){
  theMapLink.clear();
  theCounterMapLink.clear();
}

void DigiSimLinkPileUpSignals::add(const std::vector<float>& locAmpl,
			  const size_t& firstChannelWithSignal, const size_t& lastChannelWithSignal,
			  const PSimHit* hit,const int& counter){
  for (size_t iChannel=firstChannelWithSignal; iChannel<lastChannelWithSignal; ++iChannel) {
    theMapLink[iChannel].push_back(std::pair < const PSimHit*, Amplitude >(hit,Amplitude(locAmpl[iChannel])));
    theCounterMapLink[iChannel].push_back(std::make_pair(hit, counter));
  }
}

