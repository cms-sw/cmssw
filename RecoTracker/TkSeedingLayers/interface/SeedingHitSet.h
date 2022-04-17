#ifndef TkSeedingLayers_SeedingHitSet_H
#define TkSeedingLayers_SeedingHitSet_H

#include "DataFormats/TrackerRecHit2D/interface/BaseTrackerRecHit.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <vector>
#include <algorithm>

class SeedingHitSet {
public:
  using RecHit = BaseTrackerRecHit;
  using RecHitPointer = BaseTrackerRecHit *;
  using ConstRecHitPointer = BaseTrackerRecHit const *;

  static ConstRecHitPointer nullPtr() { return nullptr; }

  SeedingHitSet() = default;

  SeedingHitSet(ConstRecHitPointer one, ConstRecHitPointer two) : SeedingHitSet({one, two}) {}

  SeedingHitSet(ConstRecHitPointer one, ConstRecHitPointer two, ConstRecHitPointer three)
      : SeedingHitSet({one, two, three}) {}

  SeedingHitSet(ConstRecHitPointer one, ConstRecHitPointer two, ConstRecHitPointer three, ConstRecHitPointer four)
      : SeedingHitSet({one, two, three, four}) {}

  SeedingHitSet(const std::vector<ConstRecHitPointer> &hits) {
    if (hits.size() >= 2) {
      auto end = std::find(hits.begin(), hits.end(), nullPtr());
      auto size = std::distance(hits.begin(), end);
      if (size >= 2) {
        theRecHits.reserve(size);
        std::copy(hits.begin(), end, std::back_inserter(theRecHits));
      }
    }
  }

  SeedingHitSet(const std::initializer_list<ConstRecHitPointer> &hits) {
    if (hits.size() >= 2) {
      auto end = std::find(hits.begin(), hits.end(), nullPtr());
      auto size = std::distance(hits.begin(), end);
      if (size >= 2) {
        theRecHits.reserve(size);
        std::copy(hits.begin(), end, std::back_inserter(theRecHits));
      }
    }
  }

  ConstRecHitPointer const *data() const { return theRecHits.data(); }

  unsigned int size() const { return theRecHits.size(); }

  ConstRecHitPointer get(unsigned int i) const { return theRecHits[i]; }
  ConstRecHitPointer operator[](unsigned int i) const { return theRecHits[i]; }

protected:
  std::vector<ConstRecHitPointer> theRecHits;
};

#endif
