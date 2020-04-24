#ifndef RECOTRACKER_TRANSIENTRACKINGRECHIT_GenericProjectedRecHit2D_H
#define RECOTRACKER_TRANSIENTRACKINGRECHIT_GenericProjectedRecHit2D_H

#include "TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHit.h"
#include "TrackingTools/TransientTrackingRecHit/interface/HelpertRecHit2DLocalPos.h"
#include "TrackingTools/KalmanUpdators/interface/TrackingRecHitPropagator.h"

class GenericProjectedRecHit2D : public TransientTrackingRecHit {
public:

  GenericProjectedRecHit2D( const LocalPoint& pos, const LocalError& err,
                     const GeomDet* det, const GeomDet* originaldet,
                     const TransientTrackingRecHit::ConstRecHitPointer originalHit,
                     const TrackingRecHitPropagator* propagator);

  AlgebraicSymMatrix parametersError() const override {
    return HelpertRecHit2DLocalPos().parError( localPositionError(), *det()); 
  }

  //virtual ~GenericProjectedRecHit2D(){delete theOriginalTransientHit;}

  AlgebraicVector parameters() const override ;

  LocalPoint localPosition() const override {return theLp;}

  LocalError localPositionError() const override {return theLe;}  

  AlgebraicMatrix projectionMatrix() const override {return theProjectionMatrix;} 	

  virtual DetId geographicalId() const {return det() ? det()->geographicalId() : DetId();}

  int dimension() const override {return theDimension;}

  //this hit lays on the original surface, NOT on the projection surface
  const TrackingRecHit * hit() const override { return theOriginalTransientHit->hit(); }	

  TrackingRecHit * cloneHit() const override { return theOriginalTransientHit->cloneHit(); }	

  virtual bool isValid() const{return true;}

  std::vector<const TrackingRecHit*> recHits() const override {
  	//return theOriginalTransientHit->hit()->recHits();
	return std::vector<const TrackingRecHit*>();
  }

  std::vector<TrackingRecHit*> recHits() override {
	//should it do something different?
        return std::vector<TrackingRecHit*>();
  }

  const TrackingRecHitPropagator* propagator() const {return thePropagator;}

  bool canImproveWithTrack() const override {return true;} 
   
  const GeomDet* originalDet() const {return theOriginalDet;}

  static RecHitPointer build( const LocalPoint& pos, const LocalError& err, 
			      const GeomDet* det, const GeomDet* originaldet,
			      const TransientTrackingRecHit::ConstRecHitPointer originalHit,
			      const TrackingRecHitPropagator* propagator) {
    return RecHitPointer( new GenericProjectedRecHit2D( pos, err, det, originaldet, originalHit, propagator));
  }

  RecHitPointer clone( const TrajectoryStateOnSurface& ts, const TransientTrackingRecHitBuilder*) const;

private:

  const GeomDet* theOriginalDet;
  TransientTrackingRecHit::ConstRecHitPointer theOriginalTransientHit; 
  LocalPoint theLp;
  LocalError theLe;
  AlgebraicMatrix theProjectionMatrix;
  const TrackingRecHitPropagator* thePropagator;	 
  //const TrackingRecHit* theOriginalHit;
  int theDimension; 

  GenericProjectedRecHit2D* clone() const override {
    return new GenericProjectedRecHit2D(*this);
  }

};



#endif
