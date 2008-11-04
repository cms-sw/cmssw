#ifndef SiTrackerMRHTools_MultiTrackFilterHitCollector_h
#define SiTrackerMRHTools_MultiTrackFilterHitCollector_h

//#include "TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHit.h"
#include "RecoTracker/MeasurementDet/interface/MeasurementTracker.h"
#include "RecoTracker/SiTrackerMRHTools/interface/MultiTrajectoryMeasurement.h"
#include <vector>

class Trajectory;
class TrajectoryMeasurement;

class MultiTrackFilterHitCollector {

	public:
	MultiTrackFilterHitCollector(const MeasurementTracker* meas): theMeasurementTracker(meas){}
	
	//virtual TransientTrackingRecHit::ConstRecHitContainer recHits(const Trajectory&) const = 0;
	//virtual std::vector<TrajectoryMeasurement> recHits(const Trajectory&) const = 0;

	//added for the MTF
	virtual std::vector<TrajectoryMeasurement> recHits(const std::map<int, std::vector<TrajectoryMeasurement> >&, int, double) const = 0;

	virtual MultiTrajectoryMeasurement TSOSfinder(const std::map<int, std::vector<TrajectoryMeasurement> >& tmmap, 
						      TrajectoryMeasurement& pmeas,
						      int i) const = 0;

	const MeasurementTracker* getMeasurementTracker() const {return theMeasurementTracker;}

	void updateEvent(const edm::Event& e) const {theMeasurementTracker->update(e);}

	
	private:
	const MeasurementTracker* theMeasurementTracker;		

};

#endif
