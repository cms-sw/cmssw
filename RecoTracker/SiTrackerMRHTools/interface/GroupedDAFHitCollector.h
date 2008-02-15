#ifndef SiTrackerMRHTools_GroupedDAFHitCollector_h
#define SiTrackerMRHTools_GroupedDAFHitCollector_h
#include "TrackingTools/MeasurementDet/interface/LayerMeasurements.h"
#include "RecoTracker/SiTrackerMRHTools/interface/MultiRecHitCollector.h"
#include <vector>

class Propagator;
class MeasurementEstimator;
class SiTrackerMultiRecHitUpdator;

class GroupedDAFHitCollector :public MultiRecHitCollector {
	public:
	explicit GroupedDAFHitCollector(const MeasurementTracker* measurementTracker,
				 const SiTrackerMultiRecHitUpdator* updator,
			         const MeasurementEstimator* est,
				 const Propagator* propagator,
				 const Propagator* reversePropagator
				 ):MultiRecHitCollector(measurementTracker), theLM(measurementTracker), theUpdator(updator), theEstimator(est), thePropagator(propagator), theReversePropagator(reversePropagator){}
			

	virtual ~GroupedDAFHitCollector(){}

	//given a trajectory it returns a collection
	//of TSiTrackerMultiRecHits and InvalidTransientRecHits.
	//It tryes to build a TSiTrackerMultiRecHit for each detGroup.
	//a detGroup is a group of detectors mutually exclusive for the track's crossing point.
	//To find gouped measurements it uses the LayerMeasurements::groupedMeasurements method 
	
	virtual std::vector<TrajectoryMeasurement> recHits(const Trajectory&) const;

	const SiTrackerMultiRecHitUpdator* getUpdator() const {return theUpdator;}
	const MeasurementEstimator* getEstimator() const {return theEstimator;}
        const Propagator* getPropagator() const {return thePropagator;}
        const Propagator* getReversePropagator() const {return theReversePropagator;}

	private:
	void buildMultiRecHits(const std::vector<TrajectoryMeasurementGroup>& measgroup, std::vector<TrajectoryMeasurement>& result) const;
	
	private:
	LayerMeasurements theLM;
	const SiTrackerMultiRecHitUpdator* theUpdator;
	const MeasurementEstimator* theEstimator;
	const Propagator* thePropagator;
	const Propagator* theReversePropagator;

	

};


#endif 
