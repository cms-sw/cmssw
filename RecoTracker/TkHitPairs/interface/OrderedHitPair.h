#ifndef OrderedHitPair_H
#define OrderedHitPair_H

/** \class OrderedHitPair 
 * Associate inner and outer hits in a pair used in seed generation
 */

//#include "RecoTracker/TkHitPairs/interface/RecHitTagger.h"
//#include "DataFormats/TrackingRecHit/interface/TrackingRecHit.h"
#include "DataFormats/TrackerRecHit2D/interface/SiPixelRecHit.h"
class OrderedHitPair {
public:
  //  typedef RecHitTagger<class OuterTag{}> OuterHit;
  //  typedef RecHitTagger<class InnerTag{}> InnerHit;
  typedef SiPixelRecHit OuterHit;
  typedef SiPixelRecHit InnerHit;
  /// constructor from InnerHit and OuterHit
/*   OrderedHitPair( const InnerHit & ih, const OuterHit & oh) */
/*      : theInnerHit(ih), theOuterHit(oh)  { } */
 

   OrderedHitPair( const InnerHit * ih, const OuterHit * oh)
     : theInnerHit(ih), theOuterHit(oh)  { }
  //  const InnerHit & inner() const { return theInnerHit; }
  //  const OuterHit & outer() const { return theOuterHit; } 
  const InnerHit * inner() const { return theInnerHit; }
  const OuterHit * outer() const { return theOuterHit; } 
private:
  const  InnerHit* theInnerHit;
 const   OuterHit* theOuterHit;
};

#endif

