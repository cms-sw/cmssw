#ifndef SiTrackerMRHTools_MultiRecHitCollector_h
#define SiTrackerMRHTools_MultiRecHitCollector_h

//#include "TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHit.h"
#include "RecoTracker/MeasurementDet/interface/MeasurementTracker.h"
#include <vector>

class Trajectory;
class TrajectoryMeasurement;

class MultiRecHitCollector {

	public:
	MultiRecHitCollector(const MeasurementTracker* meas): theMeasurementTracker(meas){}
	
	//virtual TransientTrackingRecHit::ConstRecHitContainer recHits(const Trajectory&) const = 0;
	virtual std::vector<TrajectoryMeasurement> recHits(const Trajectory&) const = 0;

	const MeasurementTracker* getMeasurementTracker() const {return theMeasurementTracker;}

	void updateEvent(const edm::Event& e) const {theMeasurementTracker->update(e);}

	
	private:
	const MeasurementTracker* theMeasurementTracker;		

};

#endif

