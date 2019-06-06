#ifndef RECOTRACKER_TRANSIENTRACKINGRECHIT_TRecHit2DPosConstraint_H
#define RECOTRACKER_TRANSIENTRACKINGRECHIT_TRecHit2DPosConstraint_H

#include "DataFormats/TrackerRecHit2D/interface/trackerHitRTTI.h"
#include "DataFormats/GeometryCommonDetAlgo/interface/ErrorFrameTransformer.h"

class TRecHit2DPosConstraint final : public TrackingRecHit {
public:
  TRecHit2DPosConstraint(const LocalPoint& pos, const LocalError& err, const Surface* surface)
      : TrackingRecHit(0, int(trackerHitRTTI::notFromCluster)), pos_(pos), err_(err), surface_(surface) {}

  TRecHit2DPosConstraint(const GeomDet& idet, const LocalPoint& pos, const LocalError& err, const Surface* surface)
      : TrackingRecHit(idet, int(trackerHitRTTI::notFromCluster)), pos_(pos), err_(err) {}

  TRecHit2DPosConstraint(const TRecHit2DPosConstraint& other) = default;
  TRecHit2DPosConstraint(TRecHit2DPosConstraint&& other) = default;

  ~TRecHit2DPosConstraint() override {}

  AlgebraicVector parameters() const override {
    AlgebraicVector result(2);
    LocalPoint lp = localPosition();
    result[0] = lp.x();
    result[1] = lp.y();
    return result;
  }

  AlgebraicSymMatrix parametersError() const override {
    AlgebraicSymMatrix m(2);
    LocalError le(localPositionError());
    m[0][0] = le.xx();
    m[0][1] = le.xy();
    m[1][1] = le.yy();
    return m;
  }

  AlgebraicMatrix projectionMatrix() const override {
    AlgebraicMatrix theProjectionMatrix;
    theProjectionMatrix = AlgebraicMatrix(2, 5, 0);
    theProjectionMatrix[0][3] = 1;
    theProjectionMatrix[1][4] = 1;
    return theProjectionMatrix;
  }
  int dimension() const override { return 2; }

  LocalPoint localPosition() const override { return pos_; }
  LocalError localPositionError() const override { return err_; }

  std::vector<const TrackingRecHit*> recHits() const override { return std::vector<const TrackingRecHit*>(); }
  std::vector<TrackingRecHit*> recHits() override { return std::vector<TrackingRecHit*>(); }

  // use position?
  bool sharesInput(const TrackingRecHit*, SharedInputType) const override { return false; }

  bool canImproveWithTrack() const override { return false; }

  virtual RecHitPointer clone(const TrajectoryStateOnSurface& ts) const { return RecHitPointer(clone()); }

  static RecHitPointer build(const LocalPoint& pos, const LocalError& err, const Surface* surface) {
    return RecHitPointer(new TRecHit2DPosConstraint(pos, err, surface));
  }

  const Surface* surface() const override { return det() ? &(det()->surface()) : &(*surface_); }

  GlobalPoint globalPosition() const override { return surface()->toGlobal(localPosition()); }
  GlobalError globalPositionError() const override {
    return ErrorFrameTransformer().transform(localPositionError(), *surface());
  }
  float errorGlobalR() const override { return std::sqrt(globalPositionError().rerr(globalPosition())); }
  float errorGlobalZ() const override { return std::sqrt(globalPositionError().czz()); }
  float errorGlobalRPhi() const override {
    return globalPosition().perp() * sqrt(globalPositionError().phierr(globalPosition()));
  }

private:
  const LocalPoint pos_;
  const LocalError err_;
  ConstReferenceCountingPointer<Surface> surface_;

  TRecHit2DPosConstraint* clone() const override { return new TRecHit2DPosConstraint(*this); }
};

#endif
