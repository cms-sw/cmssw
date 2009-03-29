#ifndef TkSeedingLayers_SeedingHitSet_H
#define TkSeedingLayers_SeedingHitSet_H

#include "RecoTracker/TkSeedingLayers/interface/SeedingHit.h"
#include <vector>

class SeedingHitSet {
public:

  typedef std::vector<ctfseeding::SeedingHit> Hits;
  typedef TransientTrackingRecHit::ConstRecHitContainer RecHits;

  SeedingHitSet(const Hits & hits=Hits());
  virtual ~SeedingHitSet(){}

  unsigned int size() const { return theRecHits.size(); }
  void add(const ctfseeding::SeedingHit & aHit);
  void add(TransientTrackingRecHit::ConstRecHitPointer pHit) { theRecHits.push_back(pHit); }
  TransientTrackingRecHit::ConstRecHitPointer operator[](unsigned int i) const { return theRecHits[i]; }

  const RecHits & container() const  { return theRecHits; }

protected:
  RecHits theRecHits;
};


#endif
