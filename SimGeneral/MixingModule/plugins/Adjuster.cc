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
}
}
