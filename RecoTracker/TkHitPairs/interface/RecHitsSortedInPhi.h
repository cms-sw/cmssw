#ifndef RecHitsSortedInPhi_H
#define RecHitsSortedInPhi_H

#include "TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHit.h"

#include <vector>

/** A RecHit container sorted in phi.
 *  Provides fast access for hits in a given phi window
 *  using binary search.
 */

class RecHitsSortedInPhi {
public:

  typedef TransientTrackingRecHit::ConstRecHitPointer Hit;

  // A RecHit extension that caches the phi angle for fast access
  class HitWithPhi {
  public:
    HitWithPhi( const Hit & hit) : theHit(hit), thePhi(hit->globalPosition().phi()) {}
    HitWithPhi( float phi) : theHit(0), thePhi(phi) {}
    float phi() const {return thePhi;}
    const Hit hit() const { return theHit;}
  private:
    Hit   theHit;
    float thePhi;
  };

  struct HitLessPhi {
    bool operator()( const HitWithPhi& a, const HitWithPhi& b) { return a.phi() < b.phi(); }
  };
  typedef std::vector<HitWithPhi>::const_iterator      HitIter;
  typedef std::pair<HitIter,HitIter>            Range;

  RecHitsSortedInPhi( const std::vector<Hit>& hits);

  bool empty() const { return theHits.empty(); }

  // Returns the hits in the phi range (phi in radians).
  //  The phi interval ( phiMin, phiMax) defined as the signed path along the 
  //  trigonometric circle from the point at phiMin to the point at phiMax
  //  must be positive and smaller than pi.
  //  At least one of phiMin, phiMax must be in (-pi,pi) range.
  //  Examples of correct intervals: (-3,-2), (-4,-3), (3.1,3.2), (3,-3).
  //  Examples of WRONG intervals: (-5,-4),(3,2), (4,3), (3.2,3.1), (-3,3), (4,5).
  //  Example of use: myHits = recHitsSortedInPhi( phi-deltaPhi, phi+deltaPhi);
  //
  std::vector<Hit> hits( float phiMin, float phiMax) const;

  // Same as above but the result is allocated by the caller and passed by reference.
  //  The caller is responsible for clearing of the container "result".
  //  This interface is not nice and not safe, but is much faster, since the
  //  dominant CPU time of the "nice" method hits(phimin,phimax) is spent in
  //  memory allocation of the result!
  //
  void hits( float phiMin, float phiMax, std::vector<Hit>& result) const;

  // Fast access to the hits in the phi interval (phi in radians).
  //  The arguments must satisfy -pi <= phiMin < phiMax <= pi
  //  No check is made for this.
  //
  Range unsafeRange( float phiMin, float phiMax) const;

  std::vector<Hit> hits() const {
    std::vector<Hit> result; result.reserve(theHits.size());
    for (HitIter i=theHits.begin(); i!=theHits.end(); i++) result.push_back(i->hit());
    return result;
  }


  Range all() const {
    return Range(theHits.begin(), theHits.end());
  }

private:

  std::vector<HitWithPhi> theHits;

  static void copyResult( const Range& range, std::vector<Hit>& result) {
    result.reserve(result.size()+(range.second-range.first));
    for (HitIter i = range.first; i != range.second; i++) result.push_back( i->hit());
  }

};

#endif
