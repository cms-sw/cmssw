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
    theRecHits.reserve(hits.size());
    std::copy_n(
        hits.begin(), std::find(hits.begin(), hits.end(), nullPtr()) - hits.begin(), std::back_inserter(theRecHits));
    theSize = theRecHits.size() > 1 ? theRecHits.size() : 0;
  }

  SeedingHitSet(const std::initializer_list<ConstRecHitPointer> &hits) : SeedingHitSet() {
    theRecHits.reserve(hits.size());
    std::copy_n(
        hits.begin(), std::find(hits.begin(), hits.end(), nullPtr()) - hits.begin(), std::back_inserter(theRecHits));
    theSize = theRecHits.size() > 1 ? theRecHits.size() : 0;
  }

  ConstRecHitPointer const *data() const { return theRecHits.data(); }

  unsigned int size() const { return theSize; }

  ConstRecHitPointer get(unsigned int i) const { return theRecHits[i]; }
  ConstRecHitPointer operator[](unsigned int i) const { return theRecHits[i]; }

protected:
  std::vector<ConstRecHitPointer> theRecHits;
  unsigned int theSize = 0;
};

#endif
