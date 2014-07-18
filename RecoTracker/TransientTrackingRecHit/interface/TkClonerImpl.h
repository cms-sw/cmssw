#ifndef TKClonerImplRecHit_H
#define TKClonerImplRecHit_H

#include "DataFormats/TrackerRecHit2D/interface/TkCloner.h"


class PixelClusterParameterEstimator;
class StripClusterParameterEstimator;
class SiStripRecHitMatcher;


class TkClonerImpl final : public TkCloner {
public:
  TkClonerImpl(){}
  TkClonerImpl(const PixelClusterParameterEstimator * ipixelCPE,
	       const StripClusterParameterEstimator * istripCPE,
	       const SiStripRecHitMatcher           * iMatcher
	       ): pixelCPE(ipixelCPE), stripCPE(istripCPE), theMatcher(iMatcher){}

  using TkCloner::operator();
  virtual SiPixelRecHit * operator()(SiPixelRecHit const & hit, TrajectoryStateOnSurface const& tsos) const;
  virtual SiStripRecHit2D * operator()(SiStripRecHit2D const & hit, TrajectoryStateOnSurface const& tsos) const;
  virtual SiStripRecHit1D * operator()(SiStripRecHit1D const & hit, TrajectoryStateOnSurface const& tsos) const;
  virtual SiStripMatchedRecHit2D * operator()(SiStripMatchedRecHit2D const & hit, TrajectoryStateOnSurface const& tsos) const;
  virtual ProjectedSiStripRecHit2D * operator()(ProjectedSiStripRecHit2D const & hit, TrajectoryStateOnSurface const& tsos) const;


  using TkCloner::makeShared;
  virtual TrackingRecHit::ConstRecHitPointer makeShared(SiPixelRecHit const & hit, TrajectoryStateOnSurface const& tsos) const;
  virtual TrackingRecHit::ConstRecHitPointer makeShared(SiStripRecHit2D const & hit, TrajectoryStateOnSurface const& tsos) const;
  virtual TrackingRecHit::ConstRecHitPointer makeShared(SiStripRecHit1D const & hit, TrajectoryStateOnSurface const& tsos) const;
  virtual TrackingRecHit::ConstRecHitPointer makeShared(SiStripMatchedRecHit2D const & hit, TrajectoryStateOnSurface const& tsos) const;
  virtual TrackingRecHit::ConstRecHitPointer makeShared(ProjectedSiStripRecHit2D const & hit, TrajectoryStateOnSurface const& tsos) const;


  // project either mono or stero hit...
  ProjectedSiStripRecHit2D * project(SiStripMatchedRecHit2D const & hit, bool mono, TrajectoryStateOnSurface const& tsos) const;

private:
  const PixelClusterParameterEstimator * pixelCPE;
  const StripClusterParameterEstimator * stripCPE;
  const SiStripRecHitMatcher           * theMatcher;


};
#endif
