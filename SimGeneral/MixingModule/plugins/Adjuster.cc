#include "Adjuster.h"
#include "SimDataFormats/EncodedEventId/interface/EncodedEventId.h"

namespace edm {
namespace detail {
void doTheOffset(int bunchSpace, int bcr, std::vector<SimTrack>& simtracks, unsigned int evtNr, int vertexOffset) { 

  EncodedEventId id(bcr,evtNr);
  for (auto& item : simtracks) {
    item.setEventId(id);
    if (!item.noVertex()) {
      item.setVertexIndex(item.vertIndex() + vertexOffset);
    }
  }
}

void doTheOffset(int bunchSpace, int bcr, std::vector<SimVertex>& simvertices, unsigned int evtNr, int vertexOffset) { 

  int timeOffset = bcr * bunchSpace;
  EncodedEventId id(bcr,evtNr);
  for (auto& item : simvertices) {
    item.setEventId(id);
    item.setTof(item.position().t() + timeOffset);
  }
}

void doTheOffset(int bunchSpace, int bcr, std::vector<PSimHit>& simhits, unsigned int evtNr, int vertexOffset) { 

  int timeOffset = bcr * bunchSpace;
  EncodedEventId id(bcr,evtNr);
  for (auto& item : simhits) {
    item.setEventId(id);
    item.setTof(item.timeOfFlight() + timeOffset);
  }
}

void doTheOffset(int bunchSpace, int bcr, std::vector<PCaloHit>& calohits, unsigned int evtNr, int vertexOffset) { 

  int timeOffset = bcr * bunchSpace;
  EncodedEventId id(bcr,evtNr);
  for (auto& item : calohits) {
    item.setEventId(id);
    item.setTime(item.time() + timeOffset);
  }
}

void doTheOffset(int bunchSpace, int bcr, TrackingRecHitCollection & trackingrechits, unsigned int evtNr, int vertexOffset) {

  EncodedEventId id(bcr,evtNr);
  for (auto it = trackingrechits.begin();it!=trackingrechits.end();++it) {
    {
      SiTrackerGSMatchedRecHit2D * rechit = dynamic_cast<SiTrackerGSMatchedRecHit2D*>(&(*it));
      if(rechit){
	rechit->setEeId(id.rawId());
	continue;
      }
    }
    {
      SiTrackerGSRecHit2D * rechit = dynamic_cast<SiTrackerGSRecHit2D*>(&(*it));
      if(rechit){
	rechit->setEeId(id.rawId());
	continue;
      }
    }
  }
}

} // end namespace detail
} // end namespace edm                   
