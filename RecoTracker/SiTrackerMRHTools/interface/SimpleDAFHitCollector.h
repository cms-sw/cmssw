#ifndef SiTrackerMRHTools_SimpleDAFHitCollector_h
#define SiTrackerMRHTools_SimpleDAFHitCollector_h
#include "RecoTracker/SiTrackerMRHTools/interface/MultiRecHitCollector.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit1D.h"
#include <vector>

class Propagator;
class MeasurementEstimator;
class SiTrackerMultiRecHitUpdator;
class StripRecHit1D;

class SimpleDAFHitCollector :public MultiRecHitCollector {
	public:
	explicit SimpleDAFHitCollector(const MeasurementTracker* measurementTracker,
				 const SiTrackerMultiRecHitUpdator* updator,
			         const MeasurementEstimator* est,
				 const Propagator* propagator, bool debug
				 ):MultiRecHitCollector(measurementTracker), theUpdator(updator), theEstimator(est), thePropagator(propagator), debug_(debug){}
			

	virtual ~SimpleDAFHitCollector(){}
	
	//given a trajectory it returns a collection
        //of SiTrackerMultiRecHits and InvalidTransientRecHits.
        //For each measurement in the trajectory, measurements are looked for according to the 
        //MeasurementDet::fastMeasurements method only in the detector where the original measurement lays. 
        //If measurements are found a SiTrackerMultiRecHit is built.
	//All the components will lay on the same detector  
	
	virtual std::vector<TrajectoryMeasurement> recHits(const Trajectory&, const MeasurementTrackerEvent *theMTE) const;

	const SiTrackerMultiRecHitUpdator* getUpdator() const {return theUpdator;}
	const MeasurementEstimator* getEstimator() const {return theEstimator;}
        const Propagator* getPropagator() const {return thePropagator;}

	void Debug( const std::vector<TrajectoryMeasurement> TM ) const;

	private:
	//TransientTrackingRecHit::ConstRecHitContainer buildMultiRecHits(const std::vector<TrajectoryMeasurementGroup>& measgroup) const;
	//void buildMultiRecHits(const std::vector<TrajectoryMeasurement>& measgroup, std::vector<TrajectoryMeasurement>& result) const;
	
        TrackingRecHit * clone(BaseTrackerRecHit const & hit2D ) const {
          return new SiStripRecHit1D(hit2D.localPosition(), 
			       LocalError(hit2D.localPositionError().xx(),0.f,std::numeric_limits<float>::max()),
                               *hit2D.det(), hit2D.firstClusterRef());
        }

	private:
	const SiTrackerMultiRecHitUpdator* theUpdator;
	const MeasurementEstimator* theEstimator;
	//this actually is not used in the fastMeasurement method 	
	const Propagator* thePropagator; 
	const bool debug_;


};


#endif 
