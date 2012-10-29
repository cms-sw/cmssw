#include "SiPileUpSignals.h"
#include "SimDataFormats/TrackingHit/interface/PSimHit.h"

void SiPileUpSignals::resetSignals(){
  signal_.clear();
}

void SiPileUpSignals::add(uint32_t detID, const std::vector<double>& locAmpl,
                          const size_t& firstChannelWithSignal, const size_t& lastChannelWithSignal) {
  SignalMapType& theSignal = signal_[detID];
  for (size_t iChannel=firstChannelWithSignal; iChannel<lastChannelWithSignal; ++iChannel) {
    if(locAmpl[iChannel] != 0.0) {
      if(theSignal.find(iChannel) == theSignal.end()) {
        theSignal.insert(std::make_pair(iChannel, locAmpl[iChannel]));
      } else {
        theSignal[iChannel] += locAmpl[iChannel];
      }
    }
  }
}
