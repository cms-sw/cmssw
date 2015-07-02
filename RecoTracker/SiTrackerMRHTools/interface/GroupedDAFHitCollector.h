/** \class GroupedDAFHitCollector
  *  Returns a collection of SiTrackerMultiRecHits and InvalidRecHits given a Trajectory.
  *  Builds a SiTrackerMultiRecHit for each detGroup 
  *  (i.e. a group of detectors mutually exclusive for the track's crossing point)
  *
  *  \author tropiano, genta
  *  \review in May 2014 by brondolin 
  */

#ifndef SiTrackerMRHTools_GroupedDAFHitCollector_h
#define SiTrackerMRHTools_GroupedDAFHitCollector_h

#include "TrackingTools/MeasurementDet/interface/LayerMeasurements.h"
#include "RecoTracker/SiTrackerMRHTools/interface/MultiRecHitCollector.h"
#include <vector>

class Propagator;
class MeasurementEstimator;
class MeasurementTracker;
class SiTrackerMultiRecHitUpdator;

class GroupedDAFHitCollector :public MultiRecHitCollector {

public:
	explicit GroupedDAFHitCollector(const MeasurementTracker* measurementTracker,
				 const SiTrackerMultiRecHitUpdator* updator,
			         const MeasurementEstimator* est,
				 const Propagator* propagator,
				 const Propagator* reversePropagator, bool debug):
		MultiRecHitCollector(measurementTracker), theUpdator(updator), 
		theEstimator(est), thePropagator(propagator), theReversePropagator(reversePropagator), debug_(debug){}
			

	virtual ~GroupedDAFHitCollector(){}

	virtual std::vector<TrajectoryMeasurement> recHits(const Trajectory&, 
							   const MeasurementTrackerEvent *theMT) const override;

	const SiTrackerMultiRecHitUpdator* getUpdator() const {return theUpdator;}
	const MeasurementEstimator* getEstimator() const {return theEstimator;}
        const Propagator* getPropagator() const {return thePropagator;}
        const Propagator* getReversePropagator() const {return theReversePropagator;}

private:
	void buildMultiRecHits(const std::vector<TrajectoryMeasurementGroup>& measgroup, 
			       std::vector<TrajectoryMeasurement>& result,
			       const MeasurementTrackerEvent*& theMTE) const;
	
	const SiTrackerMultiRecHitUpdator* theUpdator;
	const MeasurementEstimator* theEstimator;
	const Propagator* thePropagator;
	const Propagator* theReversePropagator;
	const bool debug_;
};


#endif 
