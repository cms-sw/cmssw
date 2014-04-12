#ifndef RecoTracker_TkSeedingLayers_SeedComparitor_H
#define RecoTracker_TkSeedingLayers_SeedComparitor_H

/** \class SeedComparitor 
 * Base class for comparing a set of tracking seeds for compatibility.  This can
 * then be used to cleanup bad seeds.  Currently forseen are child classes that use
 * PixelStubs and Ferenc Sikler's similar objects for low Pt tracks.
 *  \author Aaron Dominguez (UNL)
 */


#include "SeedingHitSet.h"

class TrajectorySeed;
class TrackingRegion;
class TrajectoryStateOnSurface;
class FastHelix;
class GlobalTrajectoryParameters;

namespace edm { class EventSetup; }

class SeedComparitor {
 public:
  virtual ~SeedComparitor() {}
  virtual void init(const edm::EventSetup& es) = 0;
  virtual bool compatible(const SeedingHitSet  &hits, const TrackingRegion & region) const = 0;
  virtual bool compatible(const TrajectorySeed &seed) const = 0;
  virtual bool compatible(const TrajectoryStateOnSurface &,  
                          SeedingHitSet::ConstRecHitPointer hit) const = 0;
  virtual bool compatible(const SeedingHitSet  &hits, 
                          const GlobalTrajectoryParameters &helixStateAtVertex,
                          const FastHelix                  &helix,
                          const TrackingRegion & region) const = 0;
  virtual bool compatible(const SeedingHitSet  &hits, 
                          const GlobalTrajectoryParameters &straightLineStateAtVertex,
                          const TrackingRegion & region) const = 0;
};

#endif

