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
      const std::string & propagator = "PropagatorWithMaterial", double seedMomentumForBOFF = -5.0, 
      double aOriginTransverseErrorMultiplier = 1.0, double aMinOneOverPtError = 1.0) 
    : thePropagatorLabel(propagator), theBOFFMomentum(seedMomentumForBOFF), 
    theOriginTransverseErrorMultiplier(aOriginTransverseErrorMultiplier), theMinOneOverPtError(aMinOneOverPtError) { }

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
      const TransientTrackingRecHit::ConstRecHitPointer &hit) const dso_hidden;  

  
  CurvilinearTrajectoryError initialError(float sin2Theta) const  dso_hidden;
  
  void buildSeed(TrajectorySeedCollection & seedCollection,
				   const SeedingHitSet & hits,
				   const FreeTrajectoryState & fts) const  dso_hidden;

  TransientTrackingRecHit::RecHitPointer refitHit(const TransientTrackingRecHit::ConstRecHitPointer &hit, 
						  const TrajectoryStateOnSurface &state) const  dso_hidden;

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

};
#endif 
