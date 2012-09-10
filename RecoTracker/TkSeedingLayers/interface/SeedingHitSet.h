#ifndef TkSeedingLayers_SeedingHitSet_H
#define TkSeedingLayers_SeedingHitSet_H

#include "TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHit.h"

class SeedingHitSet {
public:

  typedef  TransientTrackingRecHit::ConstRecHitPointer ConstRecHitPointer;

  static ConstRecHitPointer nullPtr() { return ConstRecHitPointer();}

  SeedingHitSet(ConstRecHitPointer const & one, ConstRecHitPointer const & two) 
  // : theRecHits{{one,two,ConstRecHitPointer()}}
  {
    theRecHits[0]=one;
    theRecHits[1]=two;
  }
  SeedingHitSet(ConstRecHitPointer const & one, ConstRecHitPointer const & two, 
		ConstRecHitPointer const & three) 
  // : theRecHits{{one,two,three}},
  {
    theRecHits[0]=one;
    theRecHits[1]=two;
    theRecHits[2]=three;
  }
  
  SeedingHitSet(ConstRecHitPointer const & one, ConstRecHitPointer const & two, 
		ConstRecHitPointer const & three, ConstRecHitPointer const &four) 
  {
    theRecHits[0]=one;
    theRecHits[1]=two;
    theRecHits[2]=three;
    theRecHits[3]=four;
  }
  
  ~SeedingHitSet(){}
  
  
  unsigned int size() const { return theRecHits[3].get() ? 4 : (theRecHits[2].get() ? 3 : 2); }
  
  ConstRecHitPointer const &  get(unsigned int i) const { return theRecHits[i]; }
  ConstRecHitPointer const & operator[](unsigned int i) const { return theRecHits[i]; }
  
  
private:
  ConstRecHitPointer theRecHits[4];
};


#endif
