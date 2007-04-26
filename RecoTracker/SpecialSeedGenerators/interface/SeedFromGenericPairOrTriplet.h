#ifndef SpecialSeedGenerators_SeedFromGenericPairOrTriplet_h
#define SpecialSeedGenerators_SeedFromGenericPairOrTriplet_h
/*
Class that produces a TrajectorySeed from a generic hit pair or triplet withou the vertex constraint.
If used without B (e.g. cosmics) it checks the three hits are aligned.
If used with B it checks the initial state has a momentum greated than the threshold set in the cfg
*/
//FWK
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
//DataFormats
#include "DataFormats/TrajectorySeed/interface/TrajectorySeed.h"
#include "DataFormats/TrajectorySeed/interface/PropagationDirection.h"
#include "DataFormats/TrackingRecHit/interface/TrackingRecHit.h"
//RecoLocal
#include "RecoTracker/TkSeedingLayers/interface/SeedingHitSet.h"
//TrackingTools
#include "TrackingTools/TrajectoryState/interface/FreeTrajectoryState.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateTransform.h"
#include "TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHitBuilder.h"
#include "TrackingTools/DetLayers/interface/NavigationDirection.h"
//Geometry
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
//MagneticField
#include "MagneticField/Engine/interface/MagneticField.h"
class SeedFromGenericPairOrTriplet{
	public:
	SeedFromGenericPairOrTriplet(const edm::ParameterSet& conf,
				     const MagneticField* mf,
				     const TrackerGeometry* geom,
				     const TransientTrackingRecHitBuilder* builder,
				     bool momFromPSet);
	~SeedFromGenericPairOrTriplet(){};
	void setMomentumTo(double mom){p = mom;};

	//builds a seed from a pair or triplet. it returns a null pointer if the seed does not pass the quality filter
	TrajectorySeed*            seed(const SeedingHitSet& hits,
                                        const PropagationDirection& dir,
                                        const NavigationDirection&  seedDir,
                                        const edm::EventSetup& iSetup);
	TrajectorySeed* seedFromTriplet(const SeedingHitSet& hits,
                                        const PropagationDirection& dir,
					const NavigationDirection&  seedDir,
                                  	const edm::EventSetup& iSetup);
  	TrajectorySeed*    seedFromPair(const SeedingHitSet& hits,
                                  	const PropagationDirection& dir,
					const NavigationDirection&  seedDir);

	
	private:
	TrajectorySeed*       buildSeed(const FreeTrajectoryState* startingState,
                                  	//const TrackingRecHit* firsthit,
					std::vector<const TrackingRecHit*>& trHits,
                                  	const PropagationDirection& dir);
	//initial error estimate	
	CurvilinearTrajectoryError initialError(const TrackingRecHit* rechit);	
	//in the case of noB it returns false if 3 hist are not aligned
	//if the B is on it returns false if the initial momentum is less than p
	bool qualityFilter(const FreeTrajectoryState* startingState,
                      	   const SeedingHitSet& hits);
	const MagneticField*   magfield;
	const TrackerGeometry* tracker;	
        TrajectoryStateTransform transformer;
	const TransientTrackingRecHitBuilder* theBuilder;
	float p;
	bool setMomentum;	
};

#endif
