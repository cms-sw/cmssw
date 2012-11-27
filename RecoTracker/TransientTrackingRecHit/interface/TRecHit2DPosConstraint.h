#ifndef RECOTRACKER_TRANSIENTRACKINGRECHIT_TRecHit2DPosConstraint_H
#define RECOTRACKER_TRANSIENTRACKINGRECHIT_TRecHit2DPosConstraint_H

#include "TrackingTools/TransientTrackingRecHit/interface/TValidTrackingRecHit.h"
#include "DataFormats/GeometrySurface/interface/ReferenceCounted.h"

class GeomDetUnit;

class TRecHit2DPosConstraint GCC11_FINAL : public TValidTrackingRecHit {
public:

  virtual ~TRecHit2DPosConstraint() {}

  virtual AlgebraicVector parameters() const {
    AlgebraicVector result(2);
    LocalPoint lp = localPosition();
    result[0] = lp.x();
    result[1] = lp.y();
    return result;
  }
  
  virtual AlgebraicSymMatrix parametersError() const {
    AlgebraicSymMatrix m(2);
    LocalError le( localPositionError());
    m[0][0] = le.xx();
    m[0][1] = le.xy();
    m[1][1] = le.yy();
    return m;
  }

  virtual AlgebraicMatrix projectionMatrix() const {
    AlgebraicMatrix theProjectionMatrix;
    theProjectionMatrix = AlgebraicMatrix( 2, 5, 0);
    theProjectionMatrix[0][3] = 1;
    theProjectionMatrix[1][4] = 1;
    return theProjectionMatrix;
  }
  virtual int dimension() const {return 2;}

  virtual LocalPoint localPosition() const {return pos_;}
  virtual LocalError localPositionError() const {return err_;}

  virtual const TrackingRecHit * hit() const {return 0;}//fixme return invalid

  virtual std::vector<const TrackingRecHit*> recHits() const {
    return hit()->recHits();
  }
  virtual std::vector<TrackingRecHit*> recHits() {
    return std::vector<TrackingRecHit*>();
  }

  virtual bool canImproveWithTrack() const {return false;}

  virtual RecHitPointer clone (const TrajectoryStateOnSurface& ts) const {return clone();}

  virtual const GeomDetUnit* detUnit() const {return 0;}
  virtual const GeomDet* det() const {return 0;}

  static RecHitPointer build( const LocalPoint& pos, const LocalError& err,
			      const Surface* surface) {
    return RecHitPointer( new TRecHit2DPosConstraint( pos, err, surface));
  }

  virtual const Surface * surface() const {return &(*surface_);}

private:
  const LocalPoint pos_;
  const LocalError err_;
//   const Surface* surface_;
  ConstReferenceCountingPointer<Surface> surface_;
  /// Creates the TrackingRecHit internally, avoids redundent cloning
  TRecHit2DPosConstraint(const LocalPoint& pos,
			 const LocalError& err,
			 const Surface* surface): 
    pos_(pos),err_(err),surface_(surface) {}

  TRecHit2DPosConstraint( const TRecHit2DPosConstraint& other ):
    pos_( other.localPosition() ),err_( other.localPositionError() ), surface_((other.surface())) {}

  virtual TRecHit2DPosConstraint * clone() const {
    return new TRecHit2DPosConstraint(*this);
  }

};

#endif
