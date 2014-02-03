#include "SimGeneral/MixingModule/interface/PileUpEventPrincipal.h"

#include "SimDataFormats/CaloHit/interface/PCaloHit.h"
#include "SimDataFormats/TrackingHit/interface/PSimHit.h"
#include "SimDataFormats/Track/interface/SimTrack.h"
#include "SimDataFormats/Vertex/interface/SimVertex.h"

template <>
void PileUpEventPrincipal::adjust<SimTrack>(SimTrack& item) const {
  item.setEventId(id_);
  if(!item.noVertex()) {
    item.setVertexIndex(item.vertIndex() + vertexOffset_);
  }
}

template <>
void PileUpEventPrincipal::adjust<SimVertex>(SimVertex& item) const {
  item.setEventId(id_);
  item.setTof(item.position().t() + bunchCrossingXbunchSpace_);
}

template <>
void PileUpEventPrincipal::adjust<PSimHit>(PSimHit& item) const {
  item.setEventId(id_);
  item.setTof(item.timeOfFlight() + bunchCrossingXbunchSpace_);
}

template <>
void PileUpEventPrincipal::adjust<PCaloHit>(PCaloHit& item) const {
  item.setEventId(id_);
  item.setTime(item.time() + bunchCrossingXbunchSpace_);
}
