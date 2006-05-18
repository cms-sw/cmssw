#ifndef TransientTrackingRecHit_H
#define TransientTrackingRecHit_H

#include "DataFormats/TrackingRecHit/interface/TrackingRecHit.h"
#include "Geometry/CommonDetUnit/interface/GeomDet.h"
#include <TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h>
#include <Geometry/CommonDetAlgo/interface/AlgebraicObjects.h>
#include "DataFormats/Common/interface/OwnVector.h"

class GeomDetUnit;

class TransientTrackingRecHit : public TrackingRecHit {
public:

  typedef edm::OwnVector<const TransientTrackingRecHit>   RecHitContainer;

  explicit TransientTrackingRecHit(const GeomDet * geom) : geom_(geom) {}

  virtual TransientTrackingRecHit * clone() const = 0;

  // Extension of the TrackingRecHit interface

  /// The GomeDet* is always non-zero
  const GeomDet * det() const {return geom_;}

  /// CAUTION: the GeomDetUnit* is zero for composite hits 
  /// (matched hits in the tracker, segments in the muon).
  /// Always check this pointer before using it!
  virtual const GeomDetUnit * detUnit() const = 0;

  virtual GlobalPoint globalPosition() const ;
  virtual GlobalError globalPositionError() const ;

  /// Returns a copy of the hit with parameters and errors computed with respect 
  /// to the TrajectoryStateOnSurface given as argument.
  /// For concrete hits not capable to improve their parameters and errors
  /// this method returns an exact copy, and is equivalent to clone() without arguments.
  virtual TransientTrackingRecHit* clone (const TrajectoryStateOnSurface& ts) const {
    return clone();
  }

  /// Returns true if the clone( const TrajectoryStateOnSurface&) method returns an
  /// improved hit, false if it returns an identical copy.
  /// In order to avoid redundent copies one should call canImproveWithTrack() before 
  /// calling clone( const TrajectoryStateOnSurface&).
  virtual bool canImproveWithTrack() const {return false;}

  virtual const TrackingRecHit * hit() const = 0;
  
  /// Composite interface: returns the component hits, if any
  virtual RecHitContainer transientHits() const;

private:

  const GeomDet * geom_ ;

};

#endif

