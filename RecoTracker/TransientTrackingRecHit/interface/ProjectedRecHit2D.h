#ifndef RECOTRACKER_TRANSIENTRACKINGRECHIT_ProjectedRecHit2D_H
#define RECOTRACKER_TRANSIENTRACKINGRECHIT_ProjectedRecHit2D_H

#include "DataFormats/TrackerRecHit2D/interface/ProjectedSiStripRecHit2D.h"
#include "TrackingTools/TransientTrackingRecHit/interface/GenericTransientTrackingRecHit.h"
#include "TrackingTools/TransientTrackingRecHit/interface/HelpertRecHit2DLocalPos.h"

class SiStripRecHit2D;

class ProjectedRecHit2D : public GenericTransientTrackingRecHit {
public:

  virtual AlgebraicSymMatrix parametersError() const {
    return HelpertRecHit2DLocalPos().parError( localPositionError(), *det()); 
  }

  const GeomDetUnit* detUnit() const {return 0;}

  static RecHitPointer build( const GeomDet * geom, const ProjectedSiStripRecHit2D* rh) {
    return RecHitPointer( new ProjectedRecHit2D( geom, rh));
  }
  static RecHitPointer build( const LocalPoint& pos, const LocalError& err, const GeomDet* det,
			      const TransientTrackingRecHit& originalHit) {
    return RecHitPointer( new ProjectedRecHit2D( pos, err, det, originalHit));
  }

  RecHitPointer clone( const TrajectoryStateOnSurface& ts) const;

  //const SiStripRecHit2D* originalHit() const {return theHitData;}
  const TransientTrackingRecHit& originalHit() const {return *theOriginalHit;}

private:

  //ProjectedSiStripRecHit2D* theHitData;

  ConstRecHitPointer theOriginalHit;

  ProjectedRecHit2D( const LocalPoint& pos, const LocalError& err, const GeomDet* det,
		     const TransientTrackingRecHit& originalHit);

  ProjectedRecHit2D( const GeomDet * geom, const ProjectedSiStripRecHit2D* rh) :
    GenericTransientTrackingRecHit( geom, *rh) {}

  virtual ProjectedRecHit2D* clone() const {
    return new ProjectedRecHit2D(*this);
  }

};



#endif
