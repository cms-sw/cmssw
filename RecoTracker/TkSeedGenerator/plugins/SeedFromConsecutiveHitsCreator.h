#ifndef RecoTracker_TkSeedGenerator_SeedFromConsecutiveHitsCreator_H
#define RecoTracker_TkSeedGenerator_SeedFromConsecutiveHitsCreator_H

#include "RecoTracker/TkSeedGenerator/interface/SeedCreator.h"
#include "RecoTracker/TkSeedingLayers/interface/SeedingHitSet.h"
#include "RecoTracker/TkSeedingLayers/interface/SeedComparitor.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "TrackingTools/GeomPropagators/interface/Propagator.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"

#include "RecoTracker/TransientTrackingRecHit/interface/TkTransientTrackingRecHitBuilder.h"

#include "DataFormats/TrackerRecHit2D/interface/BaseTrackerRecHit.h"
#include "DataFormats/TrackingRecHit/interface/mayown_ptr.h"


class FreeTrajectoryState;

class dso_hidden SeedFromConsecutiveHitsCreator : public SeedCreator {
public:

  SeedFromConsecutiveHitsCreator( const edm::ParameterSet & cfg)
      : thePropagatorLabel                (cfg.getParameter<std::string>("propagator"))
      , theBOFFMomentum                   (cfg.getParameter<double>("SeedMomentumForBOFF"))
      , theOriginTransverseErrorMultiplier(cfg.getParameter<double>("OriginTransverseErrorMultiplier"))
      , theMinOneOverPtError              (cfg.getParameter<double>("MinOneOverPtError"))
      , TTRHBuilder                       (cfg.getParameter<std::string>("TTRHBuilder"))
      , mfName_(cfg.getParameter<std::string>("magneticField"))
      , forceKinematicWithRegionDirection_(cfg.getParameter<bool>("forceKinematicWithRegionDirection"))
      {}

  //dtor
  virtual ~SeedFromConsecutiveHitsCreator();

  // initialize the "event dependent state"
  virtual void init(const TrackingRegion & region,
	       const edm::EventSetup& es,
	       const SeedComparitor *filter) GCC11_FINAL;

  // make job
  // fill seedCollection with the "TrajectorySeed"
  virtual void makeSeed(TrajectorySeedCollection & seedCollection,
			const SeedingHitSet & hits) GCC11_FINAL;


private:

  virtual bool initialKinematic(GlobalTrajectoryParameters & kine,
				const SeedingHitSet & hits) const;


  bool checkHit(
      const TrajectoryStateOnSurface &tsos,
      SeedingHitSet::ConstRecHitPointer hit) const dso_hidden;


  CurvilinearTrajectoryError initialError(float sin2Theta) const  dso_hidden;

  void buildSeed(TrajectorySeedCollection & seedCollection,
		 const SeedingHitSet & hits,
		 const FreeTrajectoryState & fts) const  dso_hidden;

  SeedingHitSet::RecHitPointer
  refitHit(SeedingHitSet::ConstRecHitPointer hit,
	   const TrajectoryStateOnSurface & state) const  dso_hidden;

protected:

  std::string thePropagatorLabel;
  double theBOFFMomentum;
  double theOriginTransverseErrorMultiplier;
  double theMinOneOverPtError;

  const TrackingRegion * region = nullptr;
  const SeedComparitor *filter = nullptr;
  edm::ESHandle<TrackerGeometry> tracker;
  edm::ESHandle<Propagator>  propagatorHandle;
  edm::ESHandle<MagneticField> bfield;
  float nomField;
  bool isBOFF = false;
  std::string TTRHBuilder;
  std::string mfName_;
  bool forceKinematicWithRegionDirection_;

  TkClonerImpl cloner;


};
#endif
