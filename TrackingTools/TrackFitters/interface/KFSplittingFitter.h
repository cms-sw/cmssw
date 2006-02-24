#ifndef CD_KFSplittingFitter_H_
#define CD_KFSplittingFitter_H_

#include "TrackingTools/TrackFitters/interface/KFTrajectoryFitter.h"

/** A Kalman track fit that splits matched RecHits into individual
 *  components before fitting them.
 */

class KFSplittingFitter : public KFTrajectoryFitter {

private:

  typedef TrajectoryStateOnSurface TSOS;
  typedef FreeTrajectoryState FTS;
  typedef TrajectoryMeasurement TM;
  
public:

  KFSplittingFitter(const Propagator& aPropagator,
                    const TrajectoryStateUpdator& aUpdator,
                    const MeasurementEstimator& aEstimator) :
    KFTrajectoryFitter(aPropagator, aUpdator, aEstimator) {}

  KFSplittingFitter(const Propagator* aPropagator,
		    const TrajectoryStateUpdator* aUpdator,
		    const MeasurementEstimator* aEstimator) : 
    KFTrajectoryFitter(aPropagator, aUpdator, aEstimator) {}

  virtual KFSplittingFitter* clone() const {
    return new KFSplittingFitter(propagator(),updator(),estimator());
  }
  
  virtual vector<Trajectory> fit(const Trajectory& aTraj) const;
  virtual vector<Trajectory> fit(const TrajectorySeed& aSeed,
				 const edm::OwnVector<TransientTrackingRecHit>& hits, 
				 const TSOS& firstPredTsos) const;


};

#endif //CD_KFSplittingFitter_H_
