#ifndef OrderedHitPair_H
#define OrderedHitPair_H


#include "DataFormats/TrackingRecHit/interface/TrackingRecHit.h"

class OrderedHitPair {
public:

  typedef TrackingRecHit OuterHit;
  typedef TrackingRecHit InnerHit;

 

   OrderedHitPair( const InnerHit * ih, const OuterHit * oh)
     : theInnerHit(ih), theOuterHit(oh)  { }



  const InnerHit * inner() const { return theInnerHit; }
  const OuterHit * outer() const { return theOuterHit; } 
private:
  const  InnerHit* theInnerHit;
 const   OuterHit* theOuterHit;
};

#endif

