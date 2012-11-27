#ifndef TValidTrackingRecHit_H
#define TValidTrackingRecHit_H


#include "TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHit.h"

class GeomDetUnit;


/**
 *
 *  only for valid hits
 */
class TValidTrackingRecHit : public TransientTrackingRecHit {
public:
  
  typedef ReferenceCountingPointer<TransientTrackingRecHit>        RecHitPointer;
  typedef ConstReferenceCountingPointer<TransientTrackingRecHit>   ConstRecHitPointer;
  typedef std::vector<ConstRecHitPointer>                           RecHitContainer;
  typedef std::vector<ConstRecHitPointer>                           ConstRecHitContainer;

  template<typename... Args>
  TValidTrackingRecHit(const GeomDet * geom, Args && ...args) : 
    TransientTrackingRecHit(std::forward<Args>(args)...), geom_(geom),
    errorR_(0),errorZ_(0),errorRPhi_(0),
    hasGlobalPosition_(false), hasGlobalError_(false){}

  // to be moved in children
  TrackingRecHit * cloneHit() const { return hit()->clone();}

  // Extension of the TrackingRecHit interface
  virtual const GeomDet * det() const GCC11_FINAL {return geom_;}
  virtual const Surface * surface() const GCC11_FINAL {return &(det()->surface());}


  virtual GlobalPoint globalPosition() const GCC11_FINAL;
  virtual GlobalError globalPositionError() const GCC11_FINAL;

  float errorGlobalR() const GCC11_FINAL;
  float errorGlobalZ() const GCC11_FINAL;
  float errorGlobalRPhi() const GCC11_FINAL;


  /// Returns true if the clone( const TrajectoryStateOnSurface&) method returns an
  /// improved hit, false if it returns an identical copy.
  /// In order to avoid redundent copies one should call canImproveWithTrack() before 
  /// calling clone( const TrajectoryStateOnSurface&).
  virtual bool canImproveWithTrack() const {return false;}

 
/// cluster probability, overloaded by pixel rechits.
  virtual float clusterProbability() const { return 1.f; }

private:
  void setPositionErrors() const dso_internal;

  // this is an order that must be preserved!

   mutable GlobalPoint globalPosition_;  

  const GeomDet * geom_ ;

  // caching of some variable for fast access
  mutable float errorR_,errorZ_,errorRPhi_;
  mutable bool hasGlobalPosition_;
  mutable bool hasGlobalError_;



 
 
  // hide the clone method for ReferenceCounted. Warning: this method is still 
  // accessible via the bas class TrackingRecHit interface!
  virtual TValidTrackingRecHit * clone() const = 0;

};

#endif

