// Original author: Felice Pantaleo (CERN) <felice.pantaleo@cern.ch>
// Part of the MC-truth-graph prototype - under heavy development, not yet open
// to external contributions (see PhysicsTools/TruthInfo/README.md).

#ifndef SimCalorimetry_HGCalAssociatorProducers_DetIdRecHitMap_h
#define SimCalorimetry_HGCalAssociatorProducers_DetIdRecHitMap_h

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

namespace hgcal {

  // Maps DetId::rawId() to a global recHit index.
  //
  // The index is defined by the concatenation order used by
  // DetIdToRecHitMapProducer:
  //
  //   1. all configured HGCRecHitCollection inputs, in cfg order
  //   2. all configured reco::PFRecHitCollection inputs, in cfg order
  //
  // It is not an index into a single EDM collection unless only one collection
  // is configured.
  //
  // Build-once / lookup-many access pattern: entries are appended with add() and
  // then finalize()d (sorted) once; lookups are a binary search. This is far more
  // compact and cache-friendly than a hash map (8 B/entry, contiguous) for the
  // Phase-2 HGCal RecHit volume (millions of cells).
  class DetIdRecHitMap {
  public:
    using value_type = std::pair<uint32_t, uint32_t>;  // (detId rawId, recHit index)
    using const_iterator = std::vector<value_type>::const_iterator;

    void reserve(std::size_t n) { entries_.reserve(n); }
    [[nodiscard]] std::size_t size() const { return entries_.size(); }
    [[nodiscard]] bool empty() const { return entries_.empty(); }

    // Append a (detId -> recHit index) entry. Call finalize() once afterwards.
    void add(uint32_t detId, uint32_t recHitIndex) { entries_.emplace_back(detId, recHitIndex); }

    // Sort by detId so find() can binary-search, and drop duplicate detIds
    // (keeping the first inserted, i.e. the lowest recHit index, as the hash-map
    // build did). stable_sort preserves insertion order among equal detIds.
    // Returns the number of duplicate entries dropped.
    uint32_t finalize() {
      std::stable_sort(
          entries_.begin(), entries_.end(), [](value_type const& a, value_type const& b) { return a.first < b.first; });

      std::size_t w = 0;
      uint32_t duplicates = 0;
      for (std::size_t r = 0; r < entries_.size(); ++r) {
        if (w > 0 && entries_[w - 1].first == entries_[r].first)
          ++duplicates;
        else
          entries_[w++] = entries_[r];
      }
      entries_.resize(w);
      return duplicates;
    }

    // Lookup by detId; returns end() when absent. Requires finalize() first.
    [[nodiscard]] const_iterator find(uint32_t detId) const {
      auto it = std::lower_bound(
          entries_.begin(), entries_.end(), detId, [](value_type const& e, uint32_t key) { return e.first < key; });
      return (it != entries_.end() && it->first == detId) ? it : entries_.end();
    }

    [[nodiscard]] const_iterator begin() const { return entries_.begin(); }
    [[nodiscard]] const_iterator end() const { return entries_.end(); }

  private:
    std::vector<value_type> entries_;
  };

}  // namespace hgcal

#endif
