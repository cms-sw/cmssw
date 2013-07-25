#ifndef CD_KFSplittingFitter_H_
#define CD_KFSplittingFitter_H_

/** \class KFTrajectoryFitter
 *  A Kalman track fit that splits matched RecHits into individual
 *  components before fitting them. Ported from ORCA
 *
 *  $Date: 2007/05/09 14:17:57 $
 *  $Revision: 1.6 $
 *  \author todorov, cerati
 */

#include "TrackingTools/TrackFitters/interface/KFTrajectoryFitter.h"

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
  
  virtual std::vector<Trajectory> fit(const Trajectory& aTraj) const;
  virtual std::vector<Trajectory> fit(const TrajectorySeed& aSeed,
				      const RecHitContainer& hits, 
				      const TSOS& firstPredTsos) const;


};

#endif //CD_KFSplittingFitter_H_
