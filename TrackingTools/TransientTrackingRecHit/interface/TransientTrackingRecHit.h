#ifndef TransientTrackingRecHit_H
#define TransientTrackingRecHit_H

#include "DataFormats/TrackingRecHit/interface/TrackingRecHit.h"
#include "Geometry/CommonDetUnit/interface/GeomDet.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
#include "DataFormats/GeometrySurface/interface/Surface.h" 
#include "DataFormats/GeometrySurface/interface/ReferenceCounted.h"
#include "DataFormats/GeometrySurface/interface/BlockWipedAllocator.h"

#include "FWCore/Utilities/interface/GCC11Compatibility.h"

class GeomDetUnit;

#ifdef COUNT_HITS
void countTTRH( TrackingRecHit::Type);
#else
inline void countTTRH( TrackingRecHit::Type){}
#endif


class TransientTrackingRecHit : public TrackingRecHit, 
				public ReferenceCountedInEvent {
public:

  typedef ReferenceCountingPointer< TransientTrackingRecHit>        RecHitPointer;
  typedef ConstReferenceCountingPointer< TransientTrackingRecHit>   ConstRecHitPointer;
  typedef std::vector<ConstRecHitPointer>                           RecHitContainer;
  typedef std::vector<ConstRecHitPointer>                           ConstRecHitContainer;


  TransientTrackingRecHit(){}
  explicit TransientTrackingRecHit(TrackingRecHit::id_type id, Type type=valid) : 
    TrackingRecHit(id, type)
  {countTTRH(type);}
  
  explicit TransientTrackingRecHit(TrackingRecHit const & rh) : 
    TrackingRecHit(rh.geographicalId(), rh.type())
  {countTTRH(type());}

  virtual ~TransientTrackingRecHit(){}


  // Extension of the TrackingRecHit interface

  /// The GomeDet* can be zero for InvalidTransientRecHits and for TConstraintRecHit2Ds

  virtual const GeomDet * det() const =0;
  virtual const Surface * surface() const {return &(det()->surface());}

  /// CAUTION: the GeomDetUnit* is zero for composite hits 
  /// (matched hits in the tracker, segments in the muon).
  /// Always check this pointer before using it!
  virtual const GeomDetUnit * detUnit() const;

  virtual GlobalPoint globalPosition() const =0;
  virtual GlobalError globalPositionError() const =0;

  virtual float errorGlobalR() const=0;
  virtual float errorGlobalZ() const=0;
  virtual float errorGlobalRPhi() const=0;

  /// Returns a copy of the hit with parameters and errors computed with respect 
  /// to the TrajectoryStateOnSurface given as argument.
  /// For concrete hits not capable to improve their parameters and errors
  /// this method returns an exact copy, and is equivalent to clone() without arguments.
  virtual RecHitPointer clone (const TrajectoryStateOnSurface&) const;

  /// Returns true if the clone( const TrajectoryStateOnSurface&) method returns an
  /// improved hit, false if it returns an identical copy.
  /// In order to avoid redundent copies one should call canImproveWithTrack() before 
  /// calling clone( const TrajectoryStateOnSurface&).
  virtual bool canImproveWithTrack() const {return false;}

  virtual const TrackingRecHit * hit() const = 0;

  // clone the corresponding Persistent Hit
  virtual TrackingRecHit * cloneHit() const = 0;
  
  /// Composite interface: returns the component hits, if any
  virtual ConstRecHitContainer transientHits() const;

  
/// cluster probability, overloaded by pixel rechits.
  virtual float clusterProbability() const { return 1.f; }

private:

  // hide the clone method for ReferenceCounted. Warning: this method is still 
  // accessible via the bas class TrackingRecHit interface!
  virtual TransientTrackingRecHit * clone() const = 0;

};

#endif

