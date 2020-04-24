#ifndef RECOTRACKER_TRANSIENTRACKINGRECHIT_TRecHit1DMomConstraint_H
#define RECOTRACKER_TRANSIENTRACKINGRECHIT_TRecHit1DMomConstraint_H

#include "TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHit.h"
#include "DataFormats/GeometryCommonDetAlgo/interface/ErrorFrameTransformer.h"



class TRecHit1DMomConstraint final : public TransientTrackingRecHit {
 public:

  ~TRecHit1DMomConstraint() override {}

  AlgebraicVector parameters() const override {
    AlgebraicVector result(1);
    result[0] = charge_/fabs(mom_);
    return result;
  }
  
  AlgebraicSymMatrix parametersError() const override {
    AlgebraicSymMatrix m(1);
    m[0][0] = err_/(mom_*mom_);//parametersErrors are squared
    m[0][0] *= m[0][0];
    return m;
  }

  AlgebraicMatrix projectionMatrix() const override {
    AlgebraicMatrix theProjectionMatrix;
    theProjectionMatrix = AlgebraicMatrix( 1, 5, 0);
    theProjectionMatrix[0][0] = 1;
    return theProjectionMatrix;
  }
  int dimension() const override {return 1;}

  LocalPoint localPosition() const override {return LocalPoint(0,0,0);}
  LocalError localPositionError() const override {return LocalError(0,0,0);}

  double mom() const {return mom_;}
  double err() const {return err_;}
  int charge() const {return charge_;}


  const TrackingRecHit * hit() const override {return nullptr;}//fixme return invalid
  TrackingRecHit * cloneHit() const override { return nullptr;}

  std::vector<const TrackingRecHit*> recHits() const override { return std::vector<const TrackingRecHit*>(); }
  std::vector<TrackingRecHit*> recHits() override { return std::vector<TrackingRecHit*>(); }
  bool sharesInput( const TrackingRecHit*, SharedInputType) const override { return false;}

  bool canImproveWithTrack() const override {return false;}

  virtual RecHitPointer clone (const TrajectoryStateOnSurface& ts) const {return RecHitPointer(clone());}

  const GeomDetUnit* detUnit() const override {return nullptr;}

  static RecHitPointer build(const int charge,
			     const double mom,
			     const double err,//not sqared!!!
			     const Surface* surface) {
    return RecHitPointer( new TRecHit1DMomConstraint( charge, mom, err, surface));
  }

  const Surface * surface() const override {return surface_;}

  GlobalPoint globalPosition() const override { return GlobalPoint();  }
  GlobalError globalPositionError() const override { return GlobalError();}
  float errorGlobalR() const override { return 0;}
  float errorGlobalZ() const override { return 0; }
  float errorGlobalRPhi() const override { return 0; }


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
  
  TRecHit1DMomConstraint * clone() const override {
    return new TRecHit1DMomConstraint(*this);
  }

};

#endif
