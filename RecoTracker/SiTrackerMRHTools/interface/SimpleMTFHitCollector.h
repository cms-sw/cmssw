#ifndef SiTrackerMRHTools_SimpleMTFHitCollector_h
#define SiTrackerMRHTools_SimpleMTFHitCollector_h
#include "RecoTracker/SiTrackerMRHTools/interface/MultiRecHitCollector.h"
#include "RecoTracker/SiTrackerMRHTools/interface/MultiTrajectoryMeasurement.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
#include "RecoTracker/SiTrackerMRHTools/interface/MultiTrackFilterHitCollector.h"

#include <vector>
#include <map>

class Propagator;
class MeasurementEstimator;
class SiTrackerMultiRecHitUpdatorMTF;

typedef TrajectoryStateOnSurface TSOS;


class SimpleMTFHitCollector:public MultiTrackFilterHitCollector {
 public:
  explicit SimpleMTFHitCollector(const MeasurementTracker* measurementTracker,
				 const SiTrackerMultiRecHitUpdatorMTF* updator,
				 const MeasurementEstimator* est,
				 const Propagator* propagator
				 ):MultiTrackFilterHitCollector(measurementTracker), theUpdator(updator), theEstimator(est), thePropagator(propagator){}
  
  
  virtual ~SimpleMTFHitCollector(){}
  
  //given a trajectory it returns a collection
  //of TSiTrackerMultiRecHits and InvalidTransientRecHits.
  //For each measurement in the trajectory, measurements are looked for according to the 
  //MeasurementDet::fastMeasurements method only in the detector where the original measurement lays. 
  //If measurements are found a TSiTrackerMultiRecHit is built.
  //All the components will lay on the same detector  
  
  virtual std::vector<TrajectoryMeasurement> recHits(const std::map<int, std::vector<TrajectoryMeasurement> >& tmmap, 
						     int i,
						     double annealing=1.) const;
  
  const SiTrackerMultiRecHitUpdatorMTF* getUpdator() const {return theUpdator;}
  const MeasurementEstimator* getEstimator() const {return theEstimator;}
  const Propagator* getPropagator() const {return thePropagator;}
  
 private:
  //TransientTrackingRecHit::ConstRecHitContainer buildMultiRecHits(const std::vector<TrajectoryMeasurementGroup>& measgroup) const;
  void buildMultiRecHits(const std::vector<std::pair<int, TrajectoryMeasurement> >& measgroup, 
			 MultiTrajectoryMeasurement* mtm, 
			 std::vector<TrajectoryMeasurement>& result,
			 double annealing=1.) const;
  
  void getMeasurements(std::vector<std::pair<int, TrajectoryMeasurement> >& layermeas,
		       const std::map<int, std::vector<TrajectoryMeasurement> >& tmmap, 
		       TrajectoryMeasurement& pmeas,
		       int i) const;
  
  MultiTrajectoryMeasurement getTSOS(const std::vector<std::pair<int, TrajectoryMeasurement> >& layermeas, 
				     TransientTrackingRecHit::ConstRecHitPointer rechit,
				     int i) const;
  
  MultiTrajectoryMeasurement TSOSfinder(const std::map<int, std::vector<TrajectoryMeasurement> >& tmmap, 
					TrajectoryMeasurement& pmeas,
					int i) const;
 private:
  const SiTrackerMultiRecHitUpdatorMTF* theUpdator;
  const MeasurementEstimator* theEstimator;
  //this actually is not used in the fastMeasurement method 	
  const Propagator* thePropagator; 

  
  
};


#endif 
