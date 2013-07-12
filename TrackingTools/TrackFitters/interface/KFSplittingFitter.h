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
#include "TrackingTools/TrackFitters/interface/RecHitSplitter.h"

class KFSplittingFitter GCC11_FINAL : public TrajectoryFitter {

private:

  typedef RecHitSplitter::RecHitContainer        RecHitContainer;

  typedef TrajectoryStateOnSurface TSOS;
  typedef FreeTrajectoryState FTS;
  typedef TrajectoryMeasurement TM;
  
public:

  KFSplittingFitter(const Propagator& aPropagator,
                    const TrajectoryStateUpdator& aUpdator,
                    const MeasurementEstimator& aEstimator) :
    fitter(aPropagator, aUpdator, aEstimator) {}


  KFSplittingFitter(const Propagator* aPropagator,
		    const TrajectoryStateUpdator* aUpdator,
		    const MeasurementEstimator* aEstimator) : 
    fitter(aPropagator, aUpdator, aEstimator) {}

  virtual KFSplittingFitter* clone() const {
    return new KFSplittingFitter(fitter.propagator(),fitter.updator(),fitter.estimator());
  }
  
  Trajectory fitOne(const Trajectory& aTraj,
		    fitType type) const;
  Trajectory fitOne(const TrajectorySeed& aSeed,
		    const RecHitContainer& hits,
		    fitType type) const;
 Trajectory fitOne(const TrajectorySeed& aSeed,
		    const RecHitContainer& hits, 
		    const TSOS& firstPredTsos,
		    fitType type) const;

 private :

 void sorter(const RecHitContainer& hits, PropagationDirection dir, RecHitContainer & result) const;

 KFTrajectoryFitter fitter;

};

#endif //CD_KFSplittingFitter_H_
