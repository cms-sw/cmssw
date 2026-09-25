#ifndef SimDataFormatsTrackUniqueSimTrackId_H
#define SimDataFormatsTrackUniqueSimTrackId_H

#include "FWCore/Utilities/interface/hash_combine.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticle.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticleFwd.h"
#include <tuple>

using UniqueSimTrackId = std::pair<uint32_t, EncodedEventId>;

struct UniqueSimTrackIdHash {
  std::size_t operator()(UniqueSimTrackId const &s) const noexcept {
    return edm::hash_value(s.first, s.second.rawId());
  }
};

namespace io_v1 {
  struct SimTrackToTPMap {
    std::unordered_map<UniqueSimTrackId, TrackingParticleRef, UniqueSimTrackIdHash> mapping;
  };
}  // namespace io_v1
using SimTrackToTPMap = io_v1::SimTrackToTPMap;

#endif
