#ifndef GsfTrajectoryFitter_H_
#define GsfTrajectoryFitter_H_

#include "TrackingTools/GeomPropagators/interface/Propagator.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
#include "TrackingTools/PatternTools/interface/TrajectoryStateUpdator.h"
#include "TrackingTools/PatternTools/interface/Trajectory.h"
#include "TrackingTools/TrackFitters/interface/TrajectoryFitter.h"
#include "TrackingTools/PatternTools/interface/TrajectoryMeasurement.h"
#include "TrackingTools/DetLayers/interface/MeasurementEstimator.h"
#include "TrackingTools/DetLayers/interface/DetLayerGeometry.h"

#include <vector>

class MultiTrajectoryStateMerger;

/** A GSF fitter, similar to KFTrajectoryFitter.
 */

class GsfTrajectoryFitter  GCC11_FINAL  : public TrajectoryFitter {

private:
  typedef TrajectoryStateOnSurface TSOS;
  typedef TrajectoryMeasurement TM;

public:

  /** Constructor with explicit components for propagation, update,
   *  chi2 calculation, merging and flag for merging before / after
   *  the update (i.e. fully configured) */
  GsfTrajectoryFitter(const Propagator& aPropagator,
		      const TrajectoryStateUpdator& aUpdator,
		      const MeasurementEstimator& aEstimator,
		      const MultiTrajectoryStateMerger& aMerger,
		      const DetLayerGeometry* detLayerGeometry=0);

  virtual ~GsfTrajectoryFitter();

  Trajectory fitOne(const Trajectory& t, fitType type) const;
  Trajectory fitOne(const TrajectorySeed& aSeed,
		    const RecHitContainer& hits,
		    const TrajectoryStateOnSurface& firstPredTsos, fitType type) const;
  Trajectory fitOne(const TrajectorySeed& aSeed,
		    const RecHitContainer& hits, fitType type) const;




  const Propagator* propagator() const {return thePropagator;}
  const TrajectoryStateUpdator* updator() const {return theUpdator;}
  const MeasurementEstimator* estimator() const {return theEstimator;}
  const MultiTrajectoryStateMerger* merger() const {return theMerger;}

  virtual std::unique_ptr<TrajectoryFitter> clone() const override
  {
    return std::unique_ptr<TrajectoryFitter>(
        new GsfTrajectoryFitter(*thePropagator,
                                *theUpdator,
                                *theEstimator,
                                *theMerger,
                                theGeometry));
  }

private:
  const Propagator* thePropagator;
  const TrajectoryStateUpdator* theUpdator;
  const MeasurementEstimator* theEstimator;
  const MultiTrajectoryStateMerger* theMerger;
  const DetLayerGeometry dummyGeometry;
  const DetLayerGeometry* theGeometry;

  bool theTiming;
};

#endif //TR_GsfTrajectoryFitter_H_
