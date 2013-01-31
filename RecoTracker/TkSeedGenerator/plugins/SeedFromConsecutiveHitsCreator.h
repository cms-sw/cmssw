#ifndef RecoTracker_TkSeedGenerator_SeedFromConsecutiveHitsCreator_H
#define RecoTracker_TkSeedGenerator_SeedFromConsecutiveHitsCreator_H

#include "RecoTracker/TkSeedGenerator/interface/SeedCreator.h"
#include "RecoTracker/TkSeedingLayers/interface/SeedingHitSet.h"
#include "RecoTracker/TkSeedingLayers/interface/SeedComparitor.h"

class FreeTrajectoryState;

class SeedFromConsecutiveHitsCreator : public SeedCreator {
public:

  SeedFromConsecutiveHitsCreator( const edm::ParameterSet & cfg):
    thePropagatorLabel(cfg.getParameter<std::string>("propagator")),
    theBOFFMomentum(cfg.existsAs<double>("SeedMomentumForBOFF") ? cfg.getParameter<double>("SeedMomentumForBOFF") : 5.0),
    theOriginTransverseErrorMultiplier(cfg.existsAs<double>("OriginTransverseErrorMultiplier") ? cfg.getParameter<double>("OriginTransverseErrorMultiplier") : 1.0),
    theMinOneOverPtError(cfg.existsAs<double>("MinOneOverPtError") ? cfg.getParameter<double>("MinOneOverPtError") : 1.0)
      {}

  SeedFromConsecutiveHitsCreator( 
      const std::string & propagator = "PropagatorWithMaterial", double seedMomentumForBOFF = -5.0, double aOriginTransverseErrorMultiplier = 1.0, double aMinOneOverPtError = 1.0) 
    : thePropagatorLabel(propagator), theBOFFMomentum(seedMomentumForBOFF), 
    theOriginTransverseErrorMultiplier(aOriginTransverseErrorMultiplier), theMinOneOverPtError(aMinOneOverPtError) { }

  //dtor
  virtual ~SeedFromConsecutiveHitsCreator(){}

  virtual const TrajectorySeed * trajectorySeed(TrajectorySeedCollection & seedCollection,
						const SeedingHitSet & ordered,
						const TrackingRegion & region,
						const edm::EventSetup& es,
                                                const SeedComparitor *filter);
protected:

  virtual bool checkHit(
      const TrajectoryStateOnSurface &tsos,
      const TransientTrackingRecHit::ConstRecHitPointer &hit,
      const edm::EventSetup& es,
      const SeedComparitor *filter) const; 

  virtual GlobalTrajectoryParameters initialKinematic(
      const SeedingHitSet & hits, 
      const TrackingRegion & region, 
      const edm::EventSetup& es,
      const SeedComparitor *filter,
      bool                 &passesFilter) const;

  virtual CurvilinearTrajectoryError initialError(
      const TrackingRegion& region, 
      float sinTheta) const;

  virtual const TrajectorySeed * buildSeed(
      TrajectorySeedCollection & seedCollection,
	const SeedingHitSet & hits,
	const FreeTrajectoryState & fts,
	const edm::EventSetup& es,
        const SeedComparitor *filter) const;

  virtual TransientTrackingRecHit::RecHitPointer refitHit(
      const TransientTrackingRecHit::ConstRecHitPointer &hit, 
      const TrajectoryStateOnSurface &state) const;

protected:
    std::string thePropagatorLabel;
    double theBOFFMomentum;
    double theOriginTransverseErrorMultiplier;
    double theMinOneOverPtError;

};
#endif 
