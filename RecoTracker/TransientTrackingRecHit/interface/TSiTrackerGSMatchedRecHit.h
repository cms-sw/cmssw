#ifndef RECOTRACKER_TRANSIENTTRACKINGRECHIT_TSiTrackerGSMatchedRecHit_H
#define RECOTRACKER_TRANSIENTTRACKINGRECHIT_TSiTrackerGSMatchedRecHit_H

#include "TrackingTools/TransientTrackingRecHit/interface/GenericTransientTrackingRecHit.h"
#include<memory>

class TSiTrackerGSMatchedRecHit : public GenericTransientTrackingRecHit{
public:

  virtual const GeomDetUnit* detUnit() const;
  //const GeomDetUnit* detUnit() const {return det();}

  //const GeomDet * det() const {return geom_;}

  static TransientTrackingRecHit::RecHitPointer build( const GeomDet * geom, const TrackingRecHit * rh, 
						       float weight = 1., float annealing = 1.) {
    return TransientTrackingRecHit::RecHitPointer(new TSiTrackerGSMatchedRecHit( geom, rh, weight, annealing));
  }

  static RecHitPointer build( const GeomDet * geom, std::auto_ptr<TrackingRecHit> rh, 
			      float weight=1., float annealing=1.) {
    return RecHitPointer( new TSiTrackerGSMatchedRecHit( geom, rh, weight, annealing));
  }

  virtual RecHitPointer clone( const TrajectoryStateOnSurface& ts) const;
  //  virtual bool canImproveWithTrack() const {return (theMatcher != 0);}
  virtual bool canImproveWithTrack() const {return false;}
  virtual ConstRecHitContainer 	transientHits () const;
private:
  TSiTrackerGSMatchedRecHit (const GeomDet * geom, const TrackingRecHit * rh, 
                             float weight, float annealing) : 
  GenericTransientTrackingRecHit(geom, *rh, weight, annealing) {}

  TSiTrackerGSMatchedRecHit (const GeomDet * geom, std::auto_ptr<TrackingRecHit> rh,
                                         float weight, float annealing) : 
    GenericTransientTrackingRecHit(geom, rh.release(), weight, annealing) {}

  virtual TSiTrackerGSMatchedRecHit* clone() const {
    return new TSiTrackerGSMatchedRecHit(*this);
  }

};



#endif
