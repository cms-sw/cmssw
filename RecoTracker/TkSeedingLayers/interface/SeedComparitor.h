#ifndef RecoTracker_TkSeedingLayers_SeedComparitor_H
#define RecoTracker_TkSeedingLayers_SeedComparitor_H

/** \class SeedComparitor 
 * Base class for comparing a set of tracking seeds for compatibility.  This can
 * then be used to cleanup bad seeds.  Currently forseen are child classes that use
 * PixelStubs and Ferenc Sikler's similar objects for low Pt tracks.
 *  \author Aaron Dominguez (UNL)
 */
// #include "RecoTracker/TkSeedingLayers/interface/SeedingHitSet.h"
class TrajectorySeed;
class SeedingHitSet;
class TrackingRegion;
class TrajectoryStateOnSurface;
class FastHelix;
class GlobalTrajectoryParameters;

namespace edm { class EventSetup; }

#include "TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHit.h"

class SeedComparitor {
 public:
  virtual ~SeedComparitor() {}
//  virtual bool compatible(const SeedingHitSet &hits, const edm::EventSetup & es) = 0;
  virtual void init(const edm::EventSetup& es) = 0;
  virtual bool compatible(const SeedingHitSet  &hits, const TrackingRegion & region) = 0;
  virtual bool compatible(const TrajectorySeed &seed) const = 0;
  virtual bool compatible(const TrajectoryStateOnSurface &,
                          const TransientTrackingRecHit::ConstRecHitPointer &hit) const = 0;
  virtual bool compatible(const SeedingHitSet  &hits,
                          const GlobalTrajectoryParameters &helixStateAtVertex,
                          const FastHelix                  &helix,
                          const TrackingRegion & region) const = 0;
  virtual bool compatible(const SeedingHitSet  &hits,
                          const GlobalTrajectoryParameters &straightLineStateAtVertex,
                          const TrackingRegion & region) const = 0;
};

#endif

