#ifndef RecHitsSortedInPhi_H
#define RecHitsSortedInPhi_H

#include "DataFormats/TrackerRecHit2D/interface/BaseTrackerRecHit.h"
#include "TrackingTools/DetLayers/interface/DetLayer.h"

#include <vector>
#include<array>

/** A RecHit container sorted in phi.
 *  Provides fast access for hits in a given phi window
 *  using binary search.
 */

class RecHitsSortedInPhi {
public:

  typedef BaseTrackerRecHit const * Hit;

  // A RecHit extension that caches the phi angle for fast access
  class HitWithPhi {
  public:
    HitWithPhi( const Hit & hit) : theHit(hit), thePhi(hit->globalPosition().phi()) {}
    HitWithPhi( const Hit & hit,float phi) : theHit(hit), thePhi(phi) {}
    HitWithPhi( float phi) : theHit(0), thePhi(phi) {}
    float phi() const {return thePhi;}
    Hit const & hit() const { return theHit;}
  private:
    Hit   theHit;
    float thePhi;
  };

  struct HitLessPhi {
    bool operator()( const HitWithPhi& a, const HitWithPhi& b) { return a.phi() < b.phi(); }
  };
  typedef std::vector<HitWithPhi>::const_iterator      HitIter;
  typedef std::pair<HitIter,HitIter>            Range;

  using DoubleRange = std::array<int,4>;
  
  RecHitsSortedInPhi(const std::vector<Hit>& hits, GlobalPoint const & origin, DetLayer const * il);

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
  std::vector<Hit> hits( float phiMin, float phiMax) const;

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

  std::vector<Hit> hits() const {
    std::vector<Hit> result; result.reserve(theHits.size());
    for (HitIter i=theHits.begin(); i!=theHits.end(); i++) result.push_back(i->hit());
    return result;
  }


  Range all() const {
    return Range(theHits.begin(), theHits.end());
  }

public:
  float       phi(int i) const { return theHits[i].phi();}
  float        gv(int i) const { return isBarrel ? z[i] : gp(i).perp();}  // global v
  float        rv(int i) const { return isBarrel ? u[i] : v[i];}  // dispaced r
  GlobalPoint gp(int i) const { return GlobalPoint(x[i],y[i],z[i]);}

public:

  mutable GlobalPoint theOrigin;

  std::vector<HitWithPhi> theHits;

  DetLayer const * layer;
  bool isBarrel;

  std::vector<float> x;
  std::vector<float> y;
  std::vector<float> z;
  std::vector<float> drphi;

  // barrel: u=r, v=z, forward the opposite...
  std::vector<float> u;
  std::vector<float> v;
  std::vector<float> du;
  std::vector<float> dv;
  std::vector<float> lphi;

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
class HitDoublets {
public:
  enum layer { inner=0, outer=1};

  using Hit=RecHitsSortedInPhi::Hit;


  HitDoublets(  RecHitsSortedInPhi const & in,
		RecHitsSortedInPhi const & out) :
    layers{{&in,&out}}{}
  
  HitDoublets(HitDoublets && rh) : layers(std::move(rh.layers)), indeces(std::move(rh.indeces)){}

  void reserve(std::size_t s) { indeces.reserve(2*s);}
  std::size_t size() const { return indeces.size()/2;}
  bool empty() const { return indeces.empty();}
  void clear() { indeces.clear();}

  void add (int il, int ol) { indeces.push_back(il);indeces.push_back(ol);}

  DetLayer const * detLayer(layer l) const { return layers[l]->layer; }

  Hit const & hit(int i, layer l) const { return layers[l]->theHits[indeces[2*i+l]].hit();}
  float       phi(int i, layer l) const { return layers[l]->phi(indeces[2*i+l]);}
  float       rv(int i, layer l) const { return layers[l]->rv(indeces[2*i+l]);}
  float        z(int i, layer l) const { return layers[l]->z[indeces[2*i+l]];}
  float        x(int i, layer l) const { return layers[l]->x[indeces[2*i+l]];}
  float        y(int i, layer l) const { return layers[l]->y[indeces[2*i+l]];}
  GlobalPoint gp(int i, layer l) const { return GlobalPoint(x(i,l),y(i,l),z(i,l));}

private:

  std::array<RecHitsSortedInPhi const *,2> layers;


  std::vector<int> indeces;

};

#endif
