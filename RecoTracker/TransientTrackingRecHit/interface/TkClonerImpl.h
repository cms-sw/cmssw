#ifndef TKClonerImplRecHit_H
#define TKClonerImplRecHit_H

#include <memory>
#include "DataFormats/TrackerRecHit2D/interface/TkCloner.h"
#include "RecoLocalTracker/Phase2TrackerRecHits/interface/Phase2StripCPE.h"

class PixelClusterParameterEstimator;
class StripClusterParameterEstimator;
class SiStripRecHitMatcher;

class TkClonerImpl final : public TkCloner {
public:
  TkClonerImpl() {}
  TkClonerImpl(const PixelClusterParameterEstimator* ipixelCPE,
               const StripClusterParameterEstimator* istripCPE,
               const SiStripRecHitMatcher* iMatcher)
      : pixelCPE(ipixelCPE), stripCPE(istripCPE), theMatcher(iMatcher), phase2TrackerCPE(nullptr) {}
  TkClonerImpl(const PixelClusterParameterEstimator* ipixelCPE,
               const ClusterParameterEstimator<Phase2TrackerCluster1D>* iPhase2OTCPE)
      : pixelCPE(ipixelCPE), stripCPE(nullptr), theMatcher(nullptr), phase2TrackerCPE(iPhase2OTCPE) {}

  using TkCloner::operator();
  std::unique_ptr<SiPixelRecHit> operator()(SiPixelRecHit const& hit,
                                            TrajectoryStateOnSurface const& tsos) const override;
  std::unique_ptr<SiStripRecHit2D> operator()(SiStripRecHit2D const& hit,
                                              TrajectoryStateOnSurface const& tsos) const override;
  std::unique_ptr<SiStripRecHit1D> operator()(SiStripRecHit1D const& hit,
                                              TrajectoryStateOnSurface const& tsos) const override;
  std::unique_ptr<SiStripMatchedRecHit2D> operator()(SiStripMatchedRecHit2D const& hit,
                                                     TrajectoryStateOnSurface const& tsos) const override;
  std::unique_ptr<ProjectedSiStripRecHit2D> operator()(ProjectedSiStripRecHit2D const& hit,
                                                       TrajectoryStateOnSurface const& tsos) const override;
  std::unique_ptr<Phase2TrackerRecHit1D> operator()(Phase2TrackerRecHit1D const& hit,
                                                    TrajectoryStateOnSurface const& tsos) const override;
  std::unique_ptr<VectorHit> operator()(VectorHit const& hit, TrajectoryStateOnSurface const& tsos) const override;

  using TkCloner::makeShared;
  TrackingRecHit::ConstRecHitPointer makeShared(SiPixelRecHit const& hit,
                                                TrajectoryStateOnSurface const& tsos) const override;
  TrackingRecHit::ConstRecHitPointer makeShared(SiStripRecHit2D const& hit,
                                                TrajectoryStateOnSurface const& tsos) const override;
  TrackingRecHit::ConstRecHitPointer makeShared(SiStripRecHit1D const& hit,
                                                TrajectoryStateOnSurface const& tsos) const override;
  TrackingRecHit::ConstRecHitPointer makeShared(SiStripMatchedRecHit2D const& hit,
                                                TrajectoryStateOnSurface const& tsos) const override;
  TrackingRecHit::ConstRecHitPointer makeShared(ProjectedSiStripRecHit2D const& hit,
                                                TrajectoryStateOnSurface const& tsos) const override;
  TrackingRecHit::ConstRecHitPointer makeShared(Phase2TrackerRecHit1D const& hit,
                                                TrajectoryStateOnSurface const& tsos) const override;
  TrackingRecHit::ConstRecHitPointer makeShared(VectorHit const& hit,
                                                TrajectoryStateOnSurface const& tsos) const override;
  // project either mono or stero hit...
  std::unique_ptr<ProjectedSiStripRecHit2D> project(SiStripMatchedRecHit2D const& hit,
                                                    bool mono,
                                                    TrajectoryStateOnSurface const& tsos) const;

private:
  const PixelClusterParameterEstimator* pixelCPE;
  const StripClusterParameterEstimator* stripCPE;
  const SiStripRecHitMatcher* theMatcher;
  const ClusterParameterEstimator<Phase2TrackerCluster1D>* phase2TrackerCPE;
};
#endif
