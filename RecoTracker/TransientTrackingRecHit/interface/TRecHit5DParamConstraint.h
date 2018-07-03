#ifndef RECOTRACKER_TRANSIENTRACKINGRECHIT_TRecHit5DParamConstraint_H
#define RECOTRACKER_TRANSIENTRACKINGRECHIT_TRecHit5DParamConstraint_H

#include "TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHit.h"
#include "DataFormats/GeometryCommonDetAlgo/interface/ErrorFrameTransformer.h"
#include "DataFormats/CLHEP/interface/Migration.h"



class TRecHit5DParamConstraint final : public TransientTrackingRecHit {

private:

  TRecHit5DParamConstraint( const TrajectoryStateOnSurface& tsos ) : tsos_( tsos ) {}

  TRecHit5DParamConstraint( const TRecHit5DParamConstraint& other ) : tsos_( other.trajectoryState() ) {}

public:

  ~TRecHit5DParamConstraint() override {}

  int dimension() const override { return 5; }

  AlgebraicMatrix projectionMatrix() const override {
    AlgebraicMatrix projectionMatrix( 5, 5, 1 );
    return projectionMatrix;
  }

  AlgebraicVector parameters() const override { return asHepVector( tsos_.localParameters().vector() ); }

  AlgebraicSymMatrix parametersError() const override { return asHepMatrix( tsos_.localError().matrix() ); }

  LocalPoint localPosition() const override { return tsos_.localPosition(); }

  LocalError localPositionError() const override { return tsos_.localError().positionError(); }

  virtual int charge() const { return tsos_.charge(); }

  bool canImproveWithTrack() const override { return false; }

  const TrackingRecHit* hit() const override { return nullptr; }
  TrackingRecHit * cloneHit() const override { return nullptr;}
  
  std::vector<const TrackingRecHit*> recHits() const override { return std::vector<const TrackingRecHit*>(); }
  std::vector<TrackingRecHit*> recHits() override { return std::vector<TrackingRecHit*>(); }
  bool sharesInput( const TrackingRecHit*, SharedInputType) const override { return false;}


  const GeomDetUnit* detUnit() const override { return nullptr; }

  virtual const GeomDet* det() const { return nullptr; }

  const Surface* surface() const override { return &tsos_.surface(); }

  GlobalPoint globalPosition() const override { return  surface()->toGlobal(localPosition());}
  GlobalError globalPositionError() const override { return ErrorFrameTransformer().transform( localPositionError(), *surface() );}
  float errorGlobalR() const override { return std::sqrt(globalPositionError().rerr(globalPosition()));}
  float errorGlobalZ() const override { return std::sqrt(globalPositionError().czz()); }
  float errorGlobalRPhi() const override { return globalPosition().perp()*sqrt(globalPositionError().phierr(globalPosition())); }


  virtual TransientTrackingRecHit::RecHitPointer clone( const TrajectoryStateOnSurface& tsos ) const {
    //return new TRecHit5DParamConstraint( this->trajectoryState() );
    return RecHitPointer(new TRecHit5DParamConstraint( tsos ));
  }

  static TransientTrackingRecHit::RecHitPointer build( const TrajectoryStateOnSurface& tsos ) {
    return RecHitPointer( new TRecHit5DParamConstraint( tsos ) );
  }

private:

  const TrajectoryStateOnSurface tsos_;
  
  TRecHit5DParamConstraint* clone() const override {
    return new TRecHit5DParamConstraint( this->trajectoryState() );
  }

  const TrajectoryStateOnSurface& trajectoryState() const { return tsos_; }

};

#endif
