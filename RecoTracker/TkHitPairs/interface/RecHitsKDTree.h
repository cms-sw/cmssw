/*
 * RecHitsKDTree.h
 *
 *  Created on: Jan 28, 2016
 *      Author: fpantale
 */

#ifndef RECHITSKDTREE_H
#define RECHITSKDTREE_H
#include <vector>

#include "DataFormats/TrackerRecHit2D/interface/BaseTrackerRecHit.h"
#include "TrackingTools/DetLayers/interface/DetLayer.h"

/** A RecHit container sorted in phi.
 *  Provides fast access for hits in a given phi window
 *  using binary search.
 */

class RecHitsKDTree {
public:

  using  Hit = BaseTrackerRecHit const * ;

  // A RecHit extension that caches the phi angle for fast access
  class HitWithEtaPhi {
  public:
	  HitWithEtaPhi( const Hit & hit) : theHit(hit), thePhi(hit->globalPosition().phi()), theEta(hit->globalPosition().eta()) {}
	  HitWithEtaPhi( const Hit & hit,float eta, float phi ) : theHit(hit), theEta(eta), thePhi(phi) {}
//	  HitWithEtaPhi( float phi) : theHit(0), thePhi(phi) {}
    float phi() const {return thePhi;}
    float eta() const {return theEta;}
    Hit const & hit() const { return theHit;}
  private:
    Hit   theHit;
    float thePhi;
    float theEta;
  };

  struct HitLessPhi {
    bool operator()( const HitWithPhi& a, const HitWithPhi& b) { return a.phi() < b.phi(); }
  };

//  typedef std::vector<HitWithPhi>::const_iterator      HitIter;
//  typedef std::pair<HitIter,HitIter>            Range;

  using DoubleRange = std::array<int,4>;

  RecHitsKDTree(const std::vector<Hit>& hits, GlobalPoint const & origin, DetLayer const * il);

  bool empty() const { return theHits.empty(); }
  std::size_t size() const { return theHits.size();}


  // Returns the hits in the phi range (phi in radians).
  //  The phi interval ( phiMin, phiMax) defined as the signed path along the
  //  trigonometric circle from the point at phiMin to the point at phiMax
  //  must be positive and smaller than pi.
  //  At least one of phiMin, phiMax must be in (-pi,pi) range.
  //  Examples of correct intervals: (-3,-2), (-4,-3), (3.1,3.2), (3,-3).
  //  Examples of WRONG intervals: (-5,-4),(3,2), (4,3), (3.2,3.1), (-3,3), (4,5).
  //  Example of use: myHits = recHitsSortedInPhi( phi-deltaPhi, phi+deltaPhi);
  //

  std::vector<Hit> hits( float etaMin, float etaMax, float phiMin, float phiMax) const;
  // Same as above but the result is allocated by the caller and passed by reference.
  //  The caller is responsible for clearing of the container "result".
  //  This interface is not nice and not safe, but is much faster, since the
  //  dominant CPU time of the "nice" method hits(phimin,phimax) is spent in
  //  memory allocation of the result!
  //
  void hits( float phiMin, float phiMax, std::vector<Hit>& result) const;

  // some above, just double range of indeces..
  DoubleRange doubleRange(float phiMin, float phiMax) const;

  // Fast access to the hits in the phi interval (phi in radians).
  //  The arguments must satisfy -pi <= phiMin < phiMax <= pi
  //  No check is made for this.
  //
  Range unsafeRange( float phiMin, float phiMax) const;

public:
  float       phi(int i) const { return theHits[i].phi();}
  float        gv(int i) const { return isBarrel ? z[i] : gp(i).perp();}  // global v
  float        rv(int i) const { return isBarrel ? u[i] : v[i];}  // dispaced r
  GlobalPoint gp(int i) const { return GlobalPoint(x[i],y[i],z[i]);}

public:

  mutable GlobalPoint theOrigin;

  tbb::concurrent_vector<HitWithEtaPhi> theHits;

  DetLayer const * layer;
  bool isBarrel;

  tbb::concurrent_vector<float> x;
  tbb::concurrent_vector<float> y;
  tbb::concurrent_vector<float> z;
  tbb::concurrent_vector<float> drphi;

  // barrel: u=r, v=z, forward the opposite...
  tbb::concurrent_vector<float> u;
  tbb::concurrent_vector<float> v;
  tbb::concurrent_vector<float> du;
  tbb::concurrent_vector<float> dv;
  tbb::concurrent_vector<float> lphi;

  static void copyResult( const Range& range, std::vector<Hit>& result) {
    result.reserve(result.size()+(range.second-range.first));
    for (HitIter i = range.first; i != range.second; i++) result.push_back( i->hit());
  }

};



/*
 *   a collection of hit pairs issued by a doublet search
 * replace HitPairs as a communication mean between doublet and triplet search algos
 *
 */




#endif /* RECHITSKDTREE_H */
