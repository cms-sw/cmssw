#ifndef CD_KFTrajectorySmoother_H_
#define CD_KFTrajectorySmoother_H_

/** \class KFTrajectorySmoother
 *  A Standard Kalman smoother. The forward fit is not redone,
 *  only the backward smoothing. Ported from ORCA
 *
 *  $Date: 2007/05/09 14:17:57 $
 *  $Revision: 1.5 $
 *  \author todorov, cerati
 */

#include "TrackingTools/PatternTools/interface/TrajectorySmoother.h"
#include "TrackingTools/PatternTools/interface/TrajectoryStateUpdator.h"
#include "TrackingTools/GeomPropagators/interface/Propagator.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
#include "TrackingTools/TrajectoryState/interface/FreeTrajectoryState.h"
#include "TrackingTools/PatternTools/interface/MeasurementEstimator.h"
#include "TrackingTools/PatternTools/interface/TrajectoryMeasurement.h"

class KFTrajectorySmoother : public TrajectorySmoother {

private:

  typedef TrajectoryStateOnSurface TSOS;
  typedef FreeTrajectoryState FTS;
  typedef TrajectoryMeasurement TM;

public:

  KFTrajectorySmoother(const Propagator& aPropagator,
                       const TrajectoryStateUpdator& aUpdator,
                       const MeasurementEstimator& aEstimator,
		       float errorRescaling = 100.f,
		       int minHits = 3) :
    thePropagator(aPropagator.clone()),
    theUpdator(aUpdator.clone()),
    theEstimator(aEstimator.clone()),
    theErrorRescaling(errorRescaling),
    minHits_(minHits){}


  KFTrajectorySmoother(const Propagator* aPropagator,
		       const TrajectoryStateUpdator* aUpdator, 
		       const MeasurementEstimator* aEstimator,
		       float errorRescaling = 100.f,
		       int minHits = 3) :
    thePropagator(aPropagator->clone()),
    theUpdator(aUpdator->clone()),
    theEstimator(aEstimator->clone()),
    theErrorRescaling(errorRescaling),
    minHits_(minHits){}

  virtual ~KFTrajectorySmoother();

  virtual std::vector<Trajectory> trajectories(const Trajectory& aTraj) const;

  const Propagator* propagator() const {return thePropagator;}
  const TrajectoryStateUpdator* updator() const {return theUpdator;}
  const MeasurementEstimator* estimator() const {return theEstimator;}

  virtual KFTrajectorySmoother* clone() const{
    return new KFTrajectorySmoother(thePropagator,theUpdator,theEstimator);
  }

private:

  Propagator* thePropagator;
  const TrajectoryStateUpdator* theUpdator;
  const MeasurementEstimator* theEstimator;
  float theErrorRescaling;
  int minHits_;
};

#endif //CD_KFTrajectorySmoother_H_
