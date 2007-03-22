#ifndef RecHitsSortedInPhi_H
#define RecHitsSortedInPhi_H

#include "DataFormats/TrackingRecHit/interface/TrackingRecHit.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/CommonDetUnit/interface/GeomDet.h"

//#include "Utilities/Notification/interface/TimingReport.h"
#include <vector>

/** A RecHit container sorted in phi.
 *  Provides fast access for hits in a given phi window
 *  using binary search.
 */

class RecHitsSortedInPhi {
public:

  /** A RecHit extension that caches the phi angle for fast access
   */

  class Hit {
  public:
    Hit( const TrackingRecHit* hit,
	 const TrackerGeometry* theGeometry) 
      : theRecHit(hit)
      {
        thePhi = theGeometry->idToDet( hit->geographicalId() )->surface().toGlobal( hit->localPosition() ).phi();
      }
    Hit( float phi) : theRecHit(), thePhi( phi) {}
    float phi() const {return thePhi;}
    const TrackingRecHit* recHit() const { return theRecHit;}
  private:
    const TrackingRecHit* theRecHit;
    float thePhi;
  };

  /** Less Predicate for Hit
   */ 

  class HitLessPhi {
  public:
    bool operator()( const Hit& a, const Hit& b) {
      return a.phi() < b.phi();
    }
  };

  typedef std::vector<Hit>::const_iterator      HitIter;
  typedef std::pair<HitIter,HitIter>            Range;

  RecHitsSortedInPhi( const std::vector<const TrackingRecHit*>& hits,
		      const TrackerGeometry* theGeometry);

  /** Returns the hits in the phi range (phi in radians).
   *  The phi interval ( phiMin, phiMax) defined as the signed path along the 
   *  trigonometric circle from the point at phiMin to the point at phiMax
   *  must be positive and smaller than pi.
   *  Examples of correct intervals: (-3,-2), (-4,-3), (3.1,3.2), (3,-3).
   *  Examples of WRONG intervals: (3,2), (4,3), (3.2,3.1), (-3,3).
   *  Example of use: myHits = recHitsSortedInPhi( phi-deltaPhi, phi+deltaPhi);
   */
  std::vector<const TrackingRecHit*> hits( float phiMin, float phiMax) const;

  /** Same as above but the result is allocated by the caller and passed by reference.
   *  The caller is responsible for clearing of the container "result".
   *  This interface is not nice and not safe, but is much faster, since the
   *  dominant CPU time of the "nice" method hits(phimin,phimax) is spent in
   *  memory allocation of the result!
   *  (this is true for gcc 2.95.x, should be checked with gcc 3.2.y) 
   */
  void hits( float phiMin, float phiMax, std::vector<const TrackingRecHit*>& result) const;

  /** Fast access to the hits in the phi interval (phi in radians).
   *  The arguments must satisfy -pi <= phiMin < phiMax <= pi
   *  No check is made for this.
   */
  Range unsafeRange( float phiMin, float phiMax) const;

  std::vector<const TrackingRecHit*> hits() {
    std::vector<const TrackingRecHit*> result;
    for (std::vector<Hit>::const_iterator i=theHits.begin(); i!=theHits.end(); i++) {
      result.push_back( i->recHit());
    }
    return result;
  }

private:

  std::vector<Hit> theHits;
  /*
  static TimingReport::Item * theFillTimer;
  static TimingReport::Item * theSortTimer;
  static TimingReport::Item * theCopyResultTimer;
  static TimingReport::Item * theSearchTimer;
  static bool timingDone;

  void initTiming();
  */

  void copyResult( const Range& range, std::vector<const TrackingRecHit*>& result) const {
    //TimeMe tm2( *theCopyResultTimer, false);
    for (HitIter i = range.first; i != range.second; i++) result.push_back( i->recHit());
  }

};

#endif
