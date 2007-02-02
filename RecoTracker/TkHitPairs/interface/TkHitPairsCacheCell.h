#ifndef TkHitPairsCacheCell_H
#define TkHitPairsCacheCell_H

#include "RecoTracker/TkHitPairs/interface/TkHitPairsCachedHit.h"
#include "RecoTracker/TkMSParametrization/interface/PixelRecoRange.h"
#include <algorithm>

class TkHitPairsCacheCell {

public:
  typedef std::vector<TkHitPairsCachedHit>::const_iterator HitIter;
  typedef PixelRecoRange<HitIter> HitIterRange;

  TkHitPairsCacheCell() : theBeg(0),theEnd(0) { }
  TkHitPairsCacheCell(HitIter beg, HitIter end) : theBeg(beg), theEnd(end) { } 

  HitIterRange range() const { return HitIterRange(theBeg, theEnd ); }

  HitIterRange range_from(float phiMin) const { 
    return HitIterRange(
        lower_bound(theBeg, theEnd, phiMin, lessPhiHitVal),
        theEnd); 
  }

  HitIterRange range_upto(float phiMax) const { 
    return HitIterRange(
        theBeg, 
        upper_bound(theBeg, theEnd, phiMax, lessPhiValHit)) ; 
  }


public:

  static bool lessPhiHitHit( const TkHitPairsCachedHit * a,
      const TkHitPairsCachedHit * b) { return a->phi() < b->phi(); }
  static bool lessPhiValHit(const float & aphi, const TkHitPairsCachedHit & b) 
      { return aphi < b.phi(); }
  static bool lessPhiHitVal( const TkHitPairsCachedHit & a, const float & bphi)
      { return a.phi() < bphi; }

private:

  HitIter theBeg, theEnd;

};
#endif
