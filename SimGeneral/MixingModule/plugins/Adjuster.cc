#include "Adjuster.h"
#include "SimDataFormats/EncodedEventId/interface/EncodedEventId.h"
#include "DataFormats/TrackerRecHit2D/interface/FastTrackerRecHit.h"

namespace edm {
namespace detail {
void doTheOffset(int bunchSpace, int bcr, std::vector<SimTrack>& simtracks, unsigned int evtNr, int vertexOffset, bool wrap) { 

  EncodedEventId id(bcr,evtNr);
  for (auto& item : simtracks) {
    item.setEventId(id);
    if (!item.noVertex()) {
      item.setVertexIndex(item.vertIndex() + vertexOffset);
    }
  }
}

void doTheOffset(int bunchSpace, int bcr, std::vector<SimVertex>& simvertices, unsigned int evtNr, int vertexOffset, bool wrap) { 

  int timeOffset = bcr * bunchSpace;
  EncodedEventId id(bcr,evtNr);
  for (auto& item : simvertices) {
    item.setEventId(id);
    item.setTof(item.position().t() + timeOffset);
  }
}

void doTheOffset(int bunchSpace, int bcr, std::vector<PSimHit>& simhits, unsigned int evtNr, int vertexOffset, bool wrap) { 

  int timeOffset = bcr * bunchSpace;
  EncodedEventId id(bcr,evtNr);
  if(wrap) { // wrap time for long-lived hits into one beam crossing
    for (auto& item : simhits) {
      item.setEventId(id);

      float Tfloor = floor(item.timeOfFlight());
      float digits = item.timeOfFlight() - Tfloor;
      int remainder = int(Tfloor) % bunchSpace;
      item.setTof(float(remainder) + digits + timeOffset);
    }
  }
  else {
    for (auto& item : simhits) {
      item.setEventId(id);
      item.setTof(item.timeOfFlight() + timeOffset);
    }
  }
}

void doTheOffset(int bunchSpace, int bcr, std::vector<PCaloHit>& calohits, unsigned int evtNr, int vertexOffset, bool wrap) { 

  int timeOffset = bcr * bunchSpace;
  EncodedEventId id(bcr,evtNr);
  for (auto& item : calohits) {
    item.setEventId(id);
    item.setTime(item.time() + timeOffset);
  }
}

void doTheOffset(int bunchSpace, int bcr, TrackingRecHitCollection & trackingrechits, unsigned int evtNr, int vertexOffset, bool wrap) {

  EncodedEventId id(bcr,evtNr);
  for (auto it = trackingrechits.begin();it!=trackingrechits.end();++it) {
      if(trackerHitRTTI::isFast(*it)){
	  FastTrackerRecHit * rechit = static_cast<FastTrackerRecHit*>(&(*it));
	  rechit->setEventId(id.rawId());
      }
  }
}

} // end namespace detail
} // end namespace edm                   
