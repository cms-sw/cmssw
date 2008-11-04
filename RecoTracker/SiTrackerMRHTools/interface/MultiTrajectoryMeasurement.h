#ifndef TR_MultiTrajectoryMeasurement_H_
#define TR_MultiTrajectoryMeasurement_H_

#include "TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHit.h"

class TrackingRecHit;
class TransientTrackingRecHit;
class TrajectoryMeasurement;
class TrajectoryStateOnSurface;
class BoundSurface;
class DetLayer;

#include <vector>
#include <map>

class MultiTrajectoryMeasurement {

private:

  typedef TrajectoryMeasurement TM;
  typedef TrajectoryStateOnSurface TSOS;

public:

  MultiTrajectoryMeasurement();

  MultiTrajectoryMeasurement(TransientTrackingRecHit::ConstRecHitPointer hit, 
			     const std::map<int,TSOS>& predictions, 
			     const std::map<int,TSOS>& updates, 
			     const DetLayer*);

  MultiTrajectoryMeasurement(std::vector<TransientTrackingRecHit::ConstRecHitPointer>& hits,
			     const std::map<int,const TransientTrackingRecHit*>& multihits,
			     const std::map<int,TSOS>& predictions,
			     const std::map<int,TSOS>& updates,
			     const std::map<int,float>& estimates,
			     const DetLayer*);

  std::vector<TransientTrackingRecHit::ConstRecHitPointer> hits() const;
  std::map<int,const TransientTrackingRecHit*>& multiHits();
  std::map<int, TSOS>& filteredStates();
  std::map<int, TSOS>& predictedStates();
  std::map<int, float>& chi2s();
  const BoundSurface& surface() const;
  const DetLayer* layer() const;

private:

  std::vector<TransientTrackingRecHit::ConstRecHitPointer> theRecHits;
  std::map<int,const TransientTrackingRecHit*> theMultiHits;
  std::map<int, TSOS> theFilteredStates;
  std::map<int, TSOS> thePredictedStates;
  std::map<int, float> theChi2s;
  const DetLayer* theLayer;
};

#endif //TR_MultiTrajectoryMeasurement_H_

