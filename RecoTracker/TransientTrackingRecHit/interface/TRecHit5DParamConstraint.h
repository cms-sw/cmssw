#ifndef RECOTRACKER_TRANSIENTRACKINGRECHIT_TRecHit5DParamConstraint_H
#define RECOTRACKER_TRANSIENTRACKINGRECHIT_TRecHit5DParamConstraint_H

#include "DataFormats/TrackerRecHit2D/interface/trackerHitRTTI.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
#include "DataFormats/GeometryCommonDetAlgo/interface/ErrorFrameTransformer.h"
#include "DataFormats/CLHEP/interface/Migration.h"

class TRecHit5DParamConstraint final : public TrackingRecHit {
public:
  TRecHit5DParamConstraint(const TrajectoryStateOnSurface& tsos)
      : TrackingRecHit(0, int(trackerHitRTTI::notFromCluster)), tsos_(tsos) {}

  TRecHit5DParamConstraint(const GeomDet& idet, const TrajectoryStateOnSurface& tsos)
      : TrackingRecHit(idet, int(trackerHitRTTI::notFromCluster)), tsos_(tsos) {}

  TRecHit5DParamConstraint(const TRecHit5DParamConstraint& other) = default;
  TRecHit5DParamConstraint(TRecHit5DParamConstraint&& other) = default;

  ~TRecHit5DParamConstraint() override {}

  int dimension() const override { return 5; }

  AlgebraicMatrix projectionMatrix() const override {
    AlgebraicMatrix projectionMatrix(5, 5, 1);
    return projectionMatrix;
  }

  AlgebraicVector parameters() const override { return asHepVector(tsos_.localParameters().vector()); }

  AlgebraicSymMatrix parametersError() const override { return asHepMatrix(tsos_.localError().matrix()); }

  LocalPoint localPosition() const override { return tsos_.localPosition(); }

  LocalError localPositionError() const override { return tsos_.localError().positionError(); }

  int charge() const { return tsos_.charge(); }
  const TrajectoryStateOnSurface& trajectoryState() const { return tsos_; }

  bool canImproveWithTrack() const override { return false; }

  std::vector<const TrackingRecHit*> recHits() const override { return std::vector<const TrackingRecHit*>(); }
  std::vector<TrackingRecHit*> recHits() override { return std::vector<TrackingRecHit*>(); }

  // verify if same tsos
  bool sharesInput(const TrackingRecHit*, SharedInputType) const override { return false; }

  const Surface* surface() const override { return &tsos_.surface(); }

  GlobalPoint globalPosition() const override { return surface()->toGlobal(localPosition()); }
  GlobalError globalPositionError() const override {
    return ErrorFrameTransformer().transform(localPositionError(), *surface());
  }
  float errorGlobalR() const override { return std::sqrt(globalPositionError().rerr(globalPosition())); }
  float errorGlobalZ() const override { return std::sqrt(globalPositionError().czz()); }
  float errorGlobalRPhi() const override {
    return globalPosition().perp() * sqrt(globalPositionError().phierr(globalPosition()));
  }

  /// ????
  virtual RecHitPointer clone(const TrajectoryStateOnSurface& tsos) const {
    return RecHitPointer(new TRecHit5DParamConstraint(tsos));
  }

  static RecHitPointer build(const TrajectoryStateOnSurface& tsos) {
    return RecHitPointer(new TRecHit5DParamConstraint(tsos));
  }

private:
  const TrajectoryStateOnSurface tsos_;

  TRecHit5DParamConstraint* clone() const override { return new TRecHit5DParamConstraint(*this); }
};

#endif
