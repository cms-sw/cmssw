#ifndef CD_KFTrajectoryFitter_H_
#define CD_KFTrajectoryFitter_H_

/** \class KFTrajectoryFitter
 *  A Standard Kalman fit. Ported from ORCA
 *
 *  \author todorov, cerati
 */

#include "TrackingTools/GeomPropagators/interface/Propagator.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
#include "TrackingTools/TrajectoryState/interface/FreeTrajectoryState.h"
#include "TrackingTools/PatternTools/interface/TrajectoryStateUpdator.h"
#include "TrackingTools/PatternTools/interface/Trajectory.h"
#include "TrackingTools/TrackFitters/interface/TrajectoryFitter.h"
#include "TrackingTools/PatternTools/interface/TrajectoryMeasurement.h"
#include "TrackingTools/DetLayers/interface/MeasurementEstimator.h"
#include "TrackingTools/DetLayers/interface/DetLayerGeometry.h"

#include <memory>

class KFTrajectoryFitter GCC11_FINAL: public TrajectoryFitter {

private:

  typedef TrajectoryStateOnSurface TSOS;
  typedef FreeTrajectoryState FTS;
  typedef TrajectoryMeasurement TM;

public:


  // backward compatible (too many places it uses as such...)
  KFTrajectoryFitter(const Propagator& aPropagator,
		     const TrajectoryStateUpdator& aUpdator,
		     const MeasurementEstimator& aEstimator,
		     int minHits = 3,
		     const DetLayerGeometry* detLayerGeometry=0, 
                     TkCloner const * hc=nullptr) :
    thePropagator(aPropagator.clone()),
    theUpdator(aUpdator.clone()),
    theEstimator(aEstimator.clone()),
    theHitCloner(hc),
    theGeometry(detLayerGeometry),
    minHits_(minHits),
    owner(true){
    if(!theGeometry) theGeometry = &dummyGeometry;
    // FIXME. Why this first constructor is needed? who is using it? Can it be removed?
    // it is uses in many many places
    }


  KFTrajectoryFitter(const Propagator* aPropagator,
		     const TrajectoryStateUpdator* aUpdator,
		     const MeasurementEstimator* aEstimator,
		     int minHits = 3,
		     const DetLayerGeometry* detLayerGeometry=0,
                     TkCloner const * hc=nullptr) :
    thePropagator(aPropagator),
    theUpdator(aUpdator),
    theEstimator(aEstimator),
    theHitCloner(hc),
    theGeometry(detLayerGeometry),
    minHits_(minHits),
    owner(false){
      if(!theGeometry) theGeometry = &dummyGeometry;
    }

  ~KFTrajectoryFitter(){
    if (owner) {
      delete thePropagator;
      delete theUpdator;
      delete theEstimator;
    }
  }

  Trajectory fitOne(const Trajectory& aTraj,fitType) const;
  Trajectory fitOne(const TrajectorySeed& aSeed,
		    const RecHitContainer& hits,fitType) const;

  Trajectory fitOne(const TrajectorySeed& aSeed,
		    const RecHitContainer& hits,
		    const TSOS& firstPredTsos,fitType) const;

  const Propagator* propagator() const {return thePropagator;}
  const TrajectoryStateUpdator* updator() const {return theUpdator;}
  const MeasurementEstimator* estimator() const {return theEstimator;}

  virtual std::unique_ptr<TrajectoryFitter> clone() const override
  {
    return owner ?
        std::unique_ptr<TrajectoryFitter>(new KFTrajectoryFitter(*thePropagator,
                                                                 *theUpdator,
                                                                 *theEstimator,
                                                                 minHits_,theGeometry,theHitCloner)) :
        std::unique_ptr<TrajectoryFitter>(new KFTrajectoryFitter(thePropagator,
                                                                 theUpdator,
                                                                 theEstimator,
                                                                 minHits_,
                                                                 theGeometry,theHitCloner));
  }

 // FIXME a prototype:	final inplementaiton may differ 
  virtual void setHitCloner(TkCloner const * hc) {  theHitCloner = hc;}


private:
  KFTrajectoryFitter(KFTrajectoryFitter const&);


  static const DetLayerGeometry dummyGeometry;
  const Propagator* thePropagator;
  const TrajectoryStateUpdator* theUpdator;
  const MeasurementEstimator* theEstimator;
  TkCloner const * theHitCloner=nullptr;
  const DetLayerGeometry* theGeometry;
  int minHits_;
  bool owner;
};

#endif //CD_KFTrajectoryFitter_H_
