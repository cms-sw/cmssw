#ifndef SimDataFormatsTrackUniqueSimTrackId_H
#define SimDataFormatsTrackUniqueSimTrackId_H

#include "FWCore/Utilities/interface/hash_combine.h"

#include "SimDataFormats/EncodedEventId/interface/EncodedEventId.h"
#include <tuple>

using UniqueSimTrackId = std::pair<uint32_t, EncodedEventId>;
struct UniqueSimTrackIdHash {
  std::size_t operator()(UniqueSimTrackId const &s) const noexcept {
    return edm::hash_value(s.first, s.second.rawId());
  }
};

#endif
