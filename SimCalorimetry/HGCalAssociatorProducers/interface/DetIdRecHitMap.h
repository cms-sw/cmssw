#ifndef SimCalorimetry_HGCalAssociatorProducers_DetIdRecHitMap_h
#define SimCalorimetry_HGCalAssociatorProducers_DetIdRecHitMap_h

#include <cstdint>
#include <unordered_map>

namespace hgcal {

  // Maps DetId::rawId() to a global recHit index.
  //
  // The index is defined by the concatenation order used by
  // SimHitToRecHitMapProducer:
  //
  //   1. all configured HGCRecHitCollection inputs, in cfg order
  //   2. all configured reco::PFRecHitCollection inputs, in cfg order
  //
  // It is not an index into a single EDM collection unless only one collection
  // is configured.
  using DetIdRecHitMap = std::unordered_map<uint32_t, uint32_t>;

}  // namespace hgcal

#endif
