// Original author: Felice Pantaleo (CERN) <felice.pantaleo@cern.ch>
//
// The interaction a node belongs to, read from the packed EncodedEventId that
// ParticleData and VertexData store. One definition, because "this is the signal"
// is a statement every consumer makes and a wrong one silently labels pile-up as
// signal: the packed id of the signal is 0, which is also the default value.

#ifndef SimDataFormats_TruthInfo_interface_InteractionId_h
#define SimDataFormats_TruthInfo_interface_InteractionId_h

#include <cstdint>
#include <cstring>

#include "SimDataFormats/EncodedEventId/interface/EncodedEventId.h"

namespace truth {

  // TruthGraphProducer packs the EncodedEventId bytes into the low word of a uint64_t.
  [[nodiscard]] inline EncodedEventId decodeEventId(uint64_t packedEventId) {
    uint32_t raw = 0;
    std::memcpy(&raw, &packedEventId, sizeof(raw));
    return EncodedEventId(raw);
  }

  // The bunch crossing of the interaction, 0 for the in-time one.
  [[nodiscard]] inline int bunchCrossingOf(uint64_t packedEventId) {
    return decodeEventId(packedEventId).bunchCrossing();
  }

  // The index of the interaction inside its bunch crossing, 0 for the signal.
  [[nodiscard]] inline int eventIndexOf(uint64_t packedEventId) { return decodeEventId(packedEventId).event(); }

  // The key of a SIM object inside a merged container: a SimTrack id and a SimVertex
  // index are local to their sub-event, so only the interaction tells two of them apart.
  // A bare local id attributes the hits of a pile-up track to a signal particle.
  [[nodiscard]] inline uint64_t simObjectKey(uint64_t packedEventId, uint32_t localId) {
    return (packedEventId << 32) | static_cast<uint64_t>(localId);
  }

  // The signal interaction is the in-time one with index 0.
  [[nodiscard]] inline bool isSignalEventId(uint64_t packedEventId) {
    return bunchCrossingOf(packedEventId) == 0 && eventIndexOf(packedEventId) == 0;
  }

}  // namespace truth

#endif
