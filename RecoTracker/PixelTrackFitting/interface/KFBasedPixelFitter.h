#ifndef RecoTracker_PixelTrackFitting_KFBasedPixelFitter_h
#define RecoTracker_PixelTrackFitting_KFBasedPixelFitter_h

#include "RecoTracker/PixelTrackFitting/interface/PixelFitterBase.h"
#include "Geometry/CommonDetUnit/interface/GeomDet.h"
#include "TrackingTools/TransientTrackingRecHit/interface/TValidTrackingRecHit.h"

#include <vector>

namespace reco {
  class Track;
  class BeamSpot;
}  // namespace reco

class TransientTrackingRecHitBuilder;
class TrackerGeometry;
class MagneticField;
class TrackingRegion;
class TrackingRecHit;
class Propagator;

class KFBasedPixelFitter : public PixelFitterBase {
public:
  KFBasedPixelFitter(const Propagator *propagator,
                     const Propagator *opropagator,
                     const TransientTrackingRecHitBuilder *ttrhBuilder,
                     const TrackerGeometry *tracker,
                     const MagneticField *field,
                     const reco::BeamSpot *beamSpot);
  ~KFBasedPixelFitter() override {}

  std::unique_ptr<reco::Track> run(const std::vector<const TrackingRecHit *> &hits,
                                   const TrackingRegion &region) const override;

private:
  //this two simple classes are copied from Alignment/ReferenceTrajectories in order to avoid dependencies
  class MyBeamSpotGeomDet final : public GeomDet {
  public:
    explicit MyBeamSpotGeomDet(const ReferenceCountingPointer<BoundPlane> &plane) : GeomDet(plane) { setDetId(0); }
    ~MyBeamSpotGeomDet() override {}
    GeomDetEnumerators::SubDetector subDetector() const override { return GeomDetEnumerators::invalidDet; }
    std::vector<const GeomDet *> components() const override { return std::vector<const GeomDet *>(); }
  };
  class MyBeamSpotHit final : public TValidTrackingRecHit {
  public:
    MyBeamSpotHit(const reco::BeamSpot &beamSpot, const GeomDet *geom);
    ~MyBeamSpotHit() override {}
    LocalPoint localPosition() const override { return localPosition_; }
    LocalError localPositionError() const override { return localError_; }
    AlgebraicVector parameters() const override;
    AlgebraicSymMatrix parametersError() const override;
    int dimension() const override { return 1; }
    AlgebraicMatrix projectionMatrix() const override;
    std::vector<const TrackingRecHit *> recHits() const override { return std::vector<const TrackingRecHit *>(); }
    std::vector<TrackingRecHit *> recHits() override { return std::vector<TrackingRecHit *>(); }
    const TrackingRecHit *hit() const override { return nullptr; }

  private:
    LocalPoint localPosition_;
    LocalError localError_;
    MyBeamSpotHit *clone() const override { return new MyBeamSpotHit(*this); }
  };

  const Propagator *thePropagator;
  const Propagator *theOPropagator;
  const TransientTrackingRecHitBuilder *theTTRHBuilder;
  const TrackerGeometry *theTracker;
  const MagneticField *theField;
  const reco::BeamSpot *theBeamSpot;
};
#endif
