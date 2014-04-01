#ifndef RECOTRACKER_TRANSIENTRACKINGRECHIT_TRecHit5DParamConstraint_H
#define RECOTRACKER_TRANSIENTRACKINGRECHIT_TRecHit5DParamConstraint_H

#include "TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHit.h"
#include "DataFormats/GeometryCommonDetAlgo/interface/ErrorFrameTransformer.h"
#include "DataFormats/CLHEP/interface/Migration.h"

class GeomDetUnit;

class TRecHit5DParamConstraint GCC11_FINAL : public TransientTrackingRecHit {

private:

  TRecHit5DParamConstraint( const TrajectoryStateOnSurface& tsos ) : tsos_( tsos ) {}

  TRecHit5DParamConstraint( const TRecHit5DParamConstraint& other ) : tsos_( other.trajectoryState() ) {}

public:

  virtual ~TRecHit5DParamConstraint() {}

  virtual int dimension() const { return 5; }

  virtual AlgebraicMatrix projectionMatrix() const {
    AlgebraicMatrix projectionMatrix( 5, 5, 1 );
    return projectionMatrix;
  }

  virtual AlgebraicVector parameters() const { return asHepVector( tsos_.localParameters().vector() ); }

  virtual AlgebraicSymMatrix parametersError() const { return asHepMatrix( tsos_.localError().matrix() ); }

  virtual LocalPoint localPosition() const { return tsos_.localPosition(); }

  virtual LocalError localPositionError() const { return tsos_.localError().positionError(); }

  virtual int charge() const { return tsos_.charge(); }

  virtual bool canImproveWithTrack() const { return false; }

  virtual const TrackingRecHit* hit() const { return 0; }
  virtual TrackingRecHit * cloneHit() const { return 0;}
  
  virtual std::vector<const TrackingRecHit*> recHits() const { return std::vector<const TrackingRecHit*>(); }
  virtual std::vector<TrackingRecHit*> recHits() { return std::vector<TrackingRecHit*>(); }
  virtual bool sharesInput( const TrackingRecHit*, SharedInputType) const { return false;}


  virtual const GeomDetUnit* detUnit() const { return 0; }

  virtual const GeomDet* det() const { return 0; }

  virtual const Surface* surface() const { return &tsos_.surface(); }

  virtual GlobalPoint globalPosition() const { return  surface()->toGlobal(localPosition());}
  virtual GlobalError globalPositionError() const { return ErrorFrameTransformer().transform( localPositionError(), *surface() );}
  virtual float errorGlobalR() const { return std::sqrt(globalPositionError().rerr(globalPosition()));}
  virtual float errorGlobalZ() const { return std::sqrt(globalPositionError().czz()); }
  virtual float errorGlobalRPhi() const { return globalPosition().perp()*sqrt(globalPositionError().phierr(globalPosition())); }


  virtual TransientTrackingRecHit::RecHitPointer clone( const TrajectoryStateOnSurface& tsos ) const {
    //return new TRecHit5DParamConstraint( this->trajectoryState() );
    return RecHitPointer(new TRecHit5DParamConstraint( tsos ));
  }

  static TransientTrackingRecHit::RecHitPointer build( const TrajectoryStateOnSurface& tsos ) {
    return RecHitPointer( new TRecHit5DParamConstraint( tsos ) );
  }

private:

  const TrajectoryStateOnSurface tsos_;
  
  virtual TRecHit5DParamConstraint* clone() const {
    return new TRecHit5DParamConstraint( this->trajectoryState() );
  }

  const TrajectoryStateOnSurface& trajectoryState() const { return tsos_; }

};

#endif
