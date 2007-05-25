#ifndef TkSeedingLayers_SeedingHitSet_H
#define TkSeedingLayers_SeedingHitSet_H

#include "RecoTracker/TkSeedingLayers/interface/SeedingHit.h"
#include <vector>

class SeedingHitSet {
public:

  typedef std::vector<ctfseeding::SeedingHit> Hits;

  SeedingHitSet(const Hits & hits=Hits());
  virtual ~SeedingHitSet(){}

  void add(const ctfseeding::SeedingHit & aHit);

  const Hits & hits() const { return theHits; }

  unsigned int size() const { return theHits.size(); }

  virtual const ctfseeding::SeedingHit & operator[](unsigned int i) const { return theHits[i]; }

protected:
  Hits theHits;
};


#endif
