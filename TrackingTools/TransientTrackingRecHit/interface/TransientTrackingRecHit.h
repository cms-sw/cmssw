#ifndef TransientTrackingRecHit_H
#define TransientTrackingRecHit_H

#include "DataFormats/TrackingRecHit/interface/TrackingRecHit.h"
#include "Geometry/CommonDetUnit/interface/GeomDet.h"
#include <TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h>
#include "DataFormats/GeometrySurface/interface/Surface.h" 
//RC #include "DataFormats/Common/interface/OwnVector.h"
#include "DataFormats/GeometrySurface/interface/ReferenceCounted.h"
#include "DataFormats/GeometrySurface/interface/BlockWipedAllocator.h"


// make tis default
#define TTRH_NOGE


class GeomDetUnit;

class TransientTrackingRecHit : public TrackingRecHit, 
				public ReferenceCountedInEvent {
public:

  //RC typedef edm::OwnVector<const TransientTrackingRecHit>        RecHitContainer;

  typedef ReferenceCountingPointer< TransientTrackingRecHit>        RecHitPointer;
  typedef ConstReferenceCountingPointer< TransientTrackingRecHit>   ConstRecHitPointer;
  typedef std::vector<ConstRecHitPointer>                           RecHitContainer;
  typedef std::vector<ConstRecHitPointer>                           ConstRecHitContainer;

  explicit TransientTrackingRecHit(const GeomDet * geom=0, float weight=1., float annealing=1.) : 
    TrackingRecHit(geom ? geom->geographicalId().rawId() : 0), 
    geom_(geom), weight_(weight), annealing_(annealing),
    globalPosition_(0,0,0),
    //    globalError_(GlobalError::NullMatrix()),
    errorR_(0),errorZ_(0),errorRPhi_(0),
    hasGlobalPosition_(false), hasGlobalError_(false){}

  explicit TransientTrackingRecHit(const GeomDet * geom, DetId id, Type type=valid, float weight=1., float annealing=1. ) : 
    TrackingRecHit(id, type), 
    geom_(geom), weight_(weight), annealing_(annealing),
    globalPosition_(0,0,0),
    //    globalError_(GlobalError::NullMatrix()),
    errorR_(0),errorZ_(0),errorRPhi_(0),
    hasGlobalPosition_(false),hasGlobalError_(false){}

  explicit TransientTrackingRecHit(const GeomDet * geom, TrackingRecHit::id_type id, Type type=valid, float weight=1., float annealing=1. ) : 
    TrackingRecHit(id, type),
    geom_(geom),  weight_(weight), annealing_(annealing),
    globalPosition_(0,0,0),
    //    globalError_(GlobalError::NullMatrix()),
    errorR_(0),errorZ_(0),errorRPhi_(0),
    hasGlobalPosition_(false),hasGlobalError_(false){}
  
  explicit TransientTrackingRecHit(const GeomDet * geom, TrackingRecHit const & rh, float weight=1., float annealing=1. ) : 
    TrackingRecHit(rh.geographicalId(), rh.type()),
    geom_(geom), weight_(weight), annealing_(annealing),
    globalPosition_(0,0,0),
    //    globalError_(GlobalError::NullMatrix()),
    errorR_(0),errorZ_(0),errorRPhi_(0),
    hasGlobalPosition_(false),hasGlobalError_(false){}



  // Extension of the TrackingRecHit interface

  /// The GomeDet* can be zero for InvalidTransientRecHits and for TConstraintRecHit2Ds
  const GeomDet * det() const {return geom_;}
  virtual const Surface * surface() const {return &(geom_->surface());}

  /// CAUTION: the GeomDetUnit* is zero for composite hits 
  /// (matched hits in the tracker, segments in the muon).
  /// Always check this pointer before using it!
  virtual const GeomDetUnit * detUnit() const;

  virtual GlobalPoint globalPosition() const ;
  virtual GlobalError globalPositionError() const ;

  float errorGlobalR() const;
  float errorGlobalZ() const;
  float errorGlobalRPhi() const;

  /// Returns a copy of the hit with parameters and errors computed with respect 
  /// to the TrajectoryStateOnSurface given as argument.
  /// For concrete hits not capable to improve their parameters and errors
  /// this method returns an exact copy, and is equivalent to clone() without arguments.
  virtual RecHitPointer clone (const TrajectoryStateOnSurface& ts) const;

  /// Returns true if the clone( const TrajectoryStateOnSurface&) method returns an
  /// improved hit, false if it returns an identical copy.
  /// In order to avoid redundent copies one should call canImproveWithTrack() before 
  /// calling clone( const TrajectoryStateOnSurface&).
  virtual bool canImproveWithTrack() const {return false;}

  virtual const TrackingRecHit * hit() const = 0;
  
  /// Composite interface: returns the component hits, if any
  virtual ConstRecHitContainer transientHits() const;

  /// interface needed to set the transient hit weight and to read it back
  void setWeight(float weight){weight_ = weight;}

  float weight() const {return weight_;}
  
  /// interface needed to set and read back an annealing value that has been applied to the current hit error matrix when
  /// using it as a component for a composite rec hit (useful for the DAF)

  void setAnnealingFactor(float annealing) {annealing_ = annealing;} 

  float getAnnealingFactor() const {return annealing_;} 

  /// cluster probability, overloaded by pixel rechits.
  virtual float clusterProbability() const { return 1; }

private:
  void setPositionErrors() const;
  
  const GeomDet * geom_ ;

  float weight_;
  float annealing_;

  // caching of some variable for fast access
  mutable GlobalPoint globalPosition_;  
#ifndef TTRH_NOGE
  mutable GlobalError globalError_;
#endif
  mutable float errorR_,errorZ_,errorRPhi_;
  mutable bool hasGlobalPosition_;
  mutable bool hasGlobalError_;
 
  // hide the clone method for ReferenceCounted. Warning: this method is still 
  // accessible via the bas class TrackingRecHit interface!
  virtual TransientTrackingRecHit * clone() const = 0;

};

#endif

