#ifndef CD_KFTrajectoryFitter_H_
#define CD_KFTrajectoryFitter_H_

/** \class KFTrajectoryFitter
 *  A Standard Kalman fit. Ported from ORCA
 *
 *  $Date: 2008/10/15 09:06:48 $
 *  $Revision: 1.8 $
 *  \author todorov, cerati
 */

#include "TrackingTools/GeomPropagators/interface/Propagator.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
#include "TrackingTools/TrajectoryState/interface/FreeTrajectoryState.h"
#include "TrackingTools/PatternTools/interface/TrajectoryStateUpdator.h"
#include "TrackingTools/PatternTools/interface/Trajectory.h"
#include "TrackingTools/PatternTools/interface/TrajectoryFitter.h"
#include "TrackingTools/PatternTools/interface/TrajectoryMeasurement.h"
#include "TrackingTools/PatternTools/interface/MeasurementEstimator.h"
#include "TrackingTools/DetLayers/interface/DetLayerGeometry.h"

class KFTrajectoryFitter : public TrajectoryFitter {

private:

  typedef TrajectoryStateOnSurface TSOS;
  typedef FreeTrajectoryState FTS;
  typedef TrajectoryMeasurement TM;

public:


  KFTrajectoryFitter(const Propagator& aPropagator,
		     const TrajectoryStateUpdator& aUpdator,
		     const MeasurementEstimator& aEstimator,
		     int minHits = 3) :
    thePropagator(aPropagator.clone()),
    theUpdator(aUpdator.clone()),
    theEstimator(aEstimator.clone()),
    minHits_(minHits),
    theGeometry(0){ // to be fixed. Why this first constructor is needed? who is using it? Can it be removed?
      if(!theGeometry) theGeometry = &dummyGeometry;
    }
  
  KFTrajectoryFitter(const Propagator* aPropagator,
		     const TrajectoryStateUpdator* aUpdator,
		     const MeasurementEstimator* aEstimator,
		     int minHits = 3,
		     const DetLayerGeometry* detLayerGeometry=0) : 
    thePropagator(aPropagator->clone()),
    theUpdator(aUpdator->clone()),
    theEstimator(aEstimator->clone()),
    minHits_(minHits),
    theGeometry(detLayerGeometry){
      if(!theGeometry) theGeometry = &dummyGeometry;
    }

  virtual ~KFTrajectoryFitter(); 
  
  virtual std::vector<Trajectory> fit(const Trajectory& aTraj) const;
  virtual std::vector<Trajectory> fit(const TrajectorySeed& aSeed,
				      const RecHitContainer& hits) const;

  virtual std::vector<Trajectory> fit(const TrajectorySeed& aSeed,
				      const RecHitContainer& hits, 
				      const TSOS& firstPredTsos) const;

  const Propagator* propagator() const {return thePropagator;}
  const TrajectoryStateUpdator* updator() const {return theUpdator;}
  const MeasurementEstimator* estimator() const {return theEstimator;}
  
  virtual KFTrajectoryFitter* clone() const
  {
    return new KFTrajectoryFitter(thePropagator,theUpdator,theEstimator,minHits_,theGeometry);
  }
  
private:
  const DetLayerGeometry dummyGeometry;
  Propagator* thePropagator;
  const TrajectoryStateUpdator* theUpdator;
  const MeasurementEstimator* theEstimator;
  int minHits_;
  const DetLayerGeometry* theGeometry;
};

#endif //CD_KFTrajectoryFitter_H_
