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

  virtual AlgebraicSymMatrix parametersError() const {
    return HelpertRecHit2DLocalPos().parError( localPositionError(), *det()); 
  }

  //virtual ~GenericProjectedRecHit2D(){delete theOriginalTransientHit;}

  virtual AlgebraicVector parameters() const ;

  virtual LocalPoint localPosition() const {return theLp;}

  virtual LocalError localPositionError() const {return theLe;}  

  virtual AlgebraicMatrix projectionMatrix() const {return theProjectionMatrix;} 	

  virtual DetId geographicalId() const {return det() ? det()->geographicalId() : DetId();}

  virtual int dimension() const {return theDimension;}

  //this hit lays on the original surface, NOT on the projection surface
  virtual const TrackingRecHit * hit() const { return theOriginalTransientHit->hit(); }	

  virtual TrackingRecHit * cloneHit() const { return theOriginalTransientHit->cloneHit(); }	

  virtual bool isValid() const{return true;}

  virtual std::vector<const TrackingRecHit*> recHits() const {
  	//return theOriginalTransientHit->hit()->recHits();
	return std::vector<const TrackingRecHit*>();
  }

  virtual std::vector<TrackingRecHit*> recHits() {
	//should it do something different?
        return std::vector<TrackingRecHit*>();
  }

  const TrackingRecHitPropagator* propagator() const {return thePropagator;}

  virtual bool canImproveWithTrack() const {return true;} 
   
  const GeomDet* originalDet() const {return theOriginalDet;}

  static RecHitPointer build( const LocalPoint& pos, const LocalError& err, 
			      const GeomDet* det, const GeomDet* originaldet,
			      const TransientTrackingRecHit::ConstRecHitPointer originalHit,
			      const TrackingRecHitPropagator* propagator) {
    return RecHitPointer( new GenericProjectedRecHit2D( pos, err, det, originaldet, originalHit, propagator));
  }

  RecHitPointer clone( const TrajectoryStateOnSurface& ts) const;

private:

  const GeomDet* theOriginalDet;
  TransientTrackingRecHit::ConstRecHitPointer theOriginalTransientHit; 
  LocalPoint theLp;
  LocalError theLe;
  AlgebraicMatrix theProjectionMatrix;
  const TrackingRecHitPropagator* thePropagator;	 
  //const TrackingRecHit* theOriginalHit;
  int theDimension; 

  virtual GenericProjectedRecHit2D* clone() const {
    return new GenericProjectedRecHit2D(*this);
  }

};



#endif
