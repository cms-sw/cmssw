#include "SimTracker/SiStripDigitizer/interface/SiPileUpSignals.h"
#include "SimDataFormats/TrackingHit/interface/PSimHit.h"

void SiPileUpSignals::resetLink(){
  theMapLink.clear();
}

void SiPileUpSignals::add(const std::vector<double>& locAmpl,
			  const unsigned int& firstChannelWithSignal, const unsigned int& lastChannelWithSignal,
			  const PSimHit& hit,const int& counter){
  for (unsigned int iChannel = firstChannelWithSignal-1; iChannel<=lastChannelWithSignal-1; iChannel++){
    theMapLink[iChannel].push_back(std::pair < const PSimHit*, Amplitude >(&hit,Amplitude(locAmpl[iChannel])));
    theCounterMapLink[iChannel].push_back(std::make_pair(&hit, counter));
  }
}
