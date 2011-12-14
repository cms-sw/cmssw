#ifndef TkSeedingLayers_SeedingHitSet_H
#define TkSeedingLayers_SeedingHitSet_H

#include "TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHit.h"

class SeedingHitSet {
public:

  typedef  TransientTrackingRecHit::ConstRecHitPointer ConstRecHitPointer;

  SeedingHitSet() : m_size[0] {}
  SeedingHitSet(ConstRecHitPointer const & one, ConstRecHitPointer const & two) : theRecHits{{one,two,ConstRecHitPointer()}},m_size(2){}
  SeedingHitSet(ConstRecHitPointer const & one, ConstRecHitPointer const & two, 
		ConstRecHitPointer const & three) : theRecHits{{one,two,tree}},m_size(3){}

  ~SeedingHitSet(){}

  void add(ConstRecHitPointer pHit) { theRecHits[m_size++]=pHit; }

  unsigned int size() const { return m_size; }

  ConstRecHitPointer const &  get(unsigned int i) const { return theRecHits[i]; }
  ConstRecHitPointer const & operator[](unsigned int i) const { return theRecHits[i]; }

 
private:
  ConstRecHitPointer theRecHits[3];
  unsigned char m_size;
};


#endif
