#ifndef TkSeedingLayers_SeedingHitSet_H
#define TkSeedingLayers_SeedingHitSet_H

#include "DataFormats/TrackerRecHit2D/interface/BaseTrackerRecHit.h"

class SeedingHitSet {
public:

  using ConstRecHitPointer = BaseTrackerRecHit const *;
  using      RecHitPointer = BaseTrackerRecHit *;

  static ConstRecHitPointer nullPtr() { return nullptr;}

  SeedingHitSet() {}

  SeedingHitSet(ConstRecHitPointer one, ConstRecHitPointer two) 
  // : theRecHits{{one,two,ConstRecHitPointer()}}
  {
    theRecHits[0]=one;
    theRecHits[1]=two;
  }
  SeedingHitSet(ConstRecHitPointer  one, ConstRecHitPointer  two, 
		ConstRecHitPointer three) 
  // : theRecHits{{one,two,three}},
  {
    theRecHits[0]=one;
    theRecHits[1]=two;
    theRecHits[2]=three;
  }
  
  SeedingHitSet(ConstRecHitPointer one, ConstRecHitPointer two, 
		ConstRecHitPointer three, ConstRecHitPointer four) 
  {
    theRecHits[0]=one;
    theRecHits[1]=two;
    theRecHits[2]=three;
    theRecHits[3]=four;
  }
  
  ~SeedingHitSet(){}
  

  unsigned int size() const { return theRecHits[3] ? 4 : (theRecHits[2] ? 3 : ( theRecHits[1] ? 2 : 0 ) ); }

  ConstRecHitPointer  get(unsigned int i) const { return theRecHits[i]; }
  ConstRecHitPointer  operator[](unsigned int i) const { return theRecHits[i]; }
  
  
private:
  ConstRecHitPointer theRecHits[4];
};


#endif
