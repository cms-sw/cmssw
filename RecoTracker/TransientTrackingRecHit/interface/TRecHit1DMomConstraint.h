#ifndef RECOTRACKER_TRANSIENTRACKINGRECHIT_TRecHit1DMomConstraint_H
#define RECOTRACKER_TRANSIENTRACKINGRECHIT_TRecHit1DMomConstraint_H

#include "TrackingTools/TransientTrackingRecHit/interface/TValidTrackingRecHit.h"

class GeomDetUnit;

class TRecHit1DMomConstraint GCC11_FINAL : public TValidTrackingRecHit {
 public:

  virtual ~TRecHit1DMomConstraint() {}

  virtual AlgebraicVector parameters() const {
    AlgebraicVector result(1);
    result[0] = charge_/fabs(mom_);
    return result;
  }
  
  virtual AlgebraicSymMatrix parametersError() const {
    AlgebraicSymMatrix m(1);
    m[0][0] = err_/(mom_*mom_);//parametersErrors are squared
    m[0][0] *= m[0][0];
    return m;
  }

  virtual AlgebraicMatrix projectionMatrix() const {
    AlgebraicMatrix theProjectionMatrix;
    theProjectionMatrix = AlgebraicMatrix( 1, 5, 0);
    theProjectionMatrix[0][0] = 1;
    return theProjectionMatrix;
  }
  virtual int dimension() const {return 1;}

  virtual LocalPoint localPosition() const {return LocalPoint(0,0,0);}
  virtual LocalError localPositionError() const {return LocalError(0,0,0);}

  double mom() const {return mom_;}
  double err() const {return err_;}
  int charge() const {return charge_;}

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

  static RecHitPointer build(const int charge,
			     const double mom,
			     const double err,//not sqared!!!
			     const Surface* surface) {
    return RecHitPointer( new TRecHit1DMomConstraint( charge, mom, err, surface));
  }

  virtual const Surface * surface() const {return surface_;}

 private:
  const int    charge_;
  const double mom_;
  const double err_;
  const Surface* surface_;
  /// Creates the TrackingRecHit internally, avoids redundent cloning
  TRecHit1DMomConstraint(const int charge,
			 const double mom,
			 const double err,//notsquared
			 const Surface* surface): 
    charge_(charge),mom_(mom),err_(err),surface_(surface) {}
  
  TRecHit1DMomConstraint( const TRecHit1DMomConstraint& other ):
    charge_( other.charge() ), mom_( other.mom() ),err_( other.err() ), surface_((other.surface())) {}
  
  virtual TRecHit1DMomConstraint * clone() const {
    return new TRecHit1DMomConstraint(*this);
  }

};

#endif
