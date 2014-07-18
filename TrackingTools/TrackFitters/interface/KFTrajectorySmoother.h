#ifndef CD_KFTrajectorySmoother_H_
#define CD_KFTrajectorySmoother_H_

/** \class KFTrajectorySmoother
 *  A Standard Kalman smoother. The forward fit is not redone,
 *  only the backward smoothing. Ported from ORCA
 *
 *  \author todorov, cerati
 */

#include "TrackingTools/PatternTools/interface/TrajectorySmoother.h"
#include "TrackingTools/PatternTools/interface/TrajectoryStateUpdator.h"
#include "TrackingTools/GeomPropagators/interface/Propagator.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
#include "TrackingTools/TrajectoryState/interface/FreeTrajectoryState.h"
#include "TrackingTools/DetLayers/interface/MeasurementEstimator.h"
#include "TrackingTools/PatternTools/interface/TrajectoryMeasurement.h"
#include "TrackingTools/DetLayers/interface/DetLayerGeometry.h"

class KFTrajectorySmoother GCC11_FINAL : public TrajectorySmoother {

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
    minHits_(minHits),
    theGeometry(nullptr){ // to be fixed. Why this first constructor is needed? who is using it? Can it be removed?
      if(!theGeometry) theGeometry = &dummyGeometry;
    }


  KFTrajectorySmoother(const Propagator* aPropagator,
		       const TrajectoryStateUpdator* aUpdator, 
		       const MeasurementEstimator* aEstimator,
		       float errorRescaling = 100.f,
		       int minHits = 3,
		       const DetLayerGeometry* detLayerGeometry=nullptr,
                       TkCloner const * hc=nullptr) :
    thePropagator(aPropagator->clone()),
    theUpdator(aUpdator->clone()),
    theEstimator(aEstimator->clone()),
    theHitCloner(hc),
    theErrorRescaling(errorRescaling),
    minHits_(minHits),
    theGeometry(detLayerGeometry){
      if(!theGeometry) theGeometry = &dummyGeometry;
    }

  virtual ~KFTrajectorySmoother();

  virtual Trajectory trajectory(const Trajectory& aTraj) const;

  const Propagator* propagator() const {return thePropagator;}
  const TrajectoryStateUpdator* updator() const {return theUpdator;}
  const MeasurementEstimator* estimator() const {return theEstimator;}

  virtual KFTrajectorySmoother* clone() const{
    return new KFTrajectorySmoother(thePropagator,theUpdator,theEstimator,theErrorRescaling,minHits_,theGeometry,theHitCloner);
  }

 // FIXME a prototype:  final inplementaiton may differ
  virtual void setHitCloner(TkCloner const * hc) {  theHitCloner =	hc;}


private:
  const DetLayerGeometry dummyGeometry;
  Propagator* thePropagator;
  const TrajectoryStateUpdator* theUpdator;
  const MeasurementEstimator* theEstimator;
  TkCloner const * theHitCloner=nullptr;
  float theErrorRescaling;
  int minHits_;
  const DetLayerGeometry* theGeometry;
};

#endif //CD_KFTrajectorySmoother_H_
