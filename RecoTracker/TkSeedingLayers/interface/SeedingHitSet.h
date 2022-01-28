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

  SeedingHitSet(ConstRecHitPointer one, ConstRecHitPointer two) : theRecHits({one, two}) { setSize(); }

  SeedingHitSet(ConstRecHitPointer one, ConstRecHitPointer two, ConstRecHitPointer three)
      : theRecHits({one, two, three}) {
    setSize();
  }

  SeedingHitSet(ConstRecHitPointer one, ConstRecHitPointer two, ConstRecHitPointer three, ConstRecHitPointer four)
      : theRecHits({one, two, three, four}) {
    setSize();
  }

  SeedingHitSet(const std::vector<ConstRecHitPointer> &hits) : theRecHits(hits) { setSize(); }

  ConstRecHitPointer const *data() const { return theRecHits.data(); }

  unsigned int size() const { return theSize; }

  ConstRecHitPointer get(unsigned int i) const { return theRecHits[i]; }
  ConstRecHitPointer operator[](unsigned int i) const { return theRecHits[i]; }

protected:
  std::vector<ConstRecHitPointer> theRecHits;
  unsigned int theSize = 0;

  void setSize() {
    theSize = 0;
    while (theRecHits[++theSize] and theSize < theRecHits.size())
      ;
    theSize = theSize > 1 ? theSize : 0;
  }
};

#endif
