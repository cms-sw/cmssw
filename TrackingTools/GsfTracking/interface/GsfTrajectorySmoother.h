#ifndef GsfTrajectorySmoother_H_
#define GsfTrajectorySmoother_H_

#include "TrackingTools/PatternTools/interface/TrajectorySmoother.h"
#include "TrackingTools/PatternTools/interface/TrajectoryStateUpdator.h"
#include "TrackingTools/GeomPropagators/interface/Propagator.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
#include "TrackingTools/DetLayers/interface/MeasurementEstimator.h"
#include "TrackingTools/PatternTools/interface/TrajectoryMeasurement.h"
#include "TrackingTools/GsfTracking/interface/GsfPropagatorWithMaterial.h"
#include "TrackingTools/GsfTools/interface/GsfPropagatorAdapter.h"
#include "TrackingTools/GsfTracking/interface/FullConvolutionWithMaterial.h"
#include "TrackingTools/DetLayers/interface/DetLayerGeometry.h"

class MultiTrajectoryStateMerger;

/** A GSF smoother, similar to KFTrajectorySmoother, but (for
 *  testing purposes) without combination with the forward fit. 
 */

class GsfTrajectorySmoother GCC11_FINAL : public TrajectorySmoother {

private:

  typedef TrajectoryStateOnSurface TSOS;
  typedef TrajectoryMeasurement TM;

public:

  /** Constructor with explicit components for propagation, update,
   *  chi2 calculation, merging and flag for merging before / after
   *  the update (i.e. fully configured). It clones the algorithms. */
  GsfTrajectorySmoother(const GsfPropagatorWithMaterial& aPropagator,
			const TrajectoryStateUpdator& aUpdator,
			const MeasurementEstimator& aEstimator,
			const MultiTrajectoryStateMerger& merger,
			float errorRescaling,
			const bool materialBeforeUpdate = true,
			const DetLayerGeometry* detLayerGeometry=0);

  virtual ~GsfTrajectorySmoother();

  virtual Trajectory trajectory(const Trajectory& aTraj) const;

  /** propagator used (full propagator, if material effects are
   * applied before the update, otherwise purely geometrical part)
   */
  const Propagator* propagator() const {
    if ( thePropagator) return thePropagator;
    else  return theGeomPropagator;
  }
  const TrajectoryStateUpdator* updator() const {return theUpdator;}
  const MeasurementEstimator* estimator() const {return theEstimator;}

  virtual GsfTrajectorySmoother* clone() const
  {
    return new GsfTrajectorySmoother(*thePropagator,*theUpdator,*theEstimator,
				     *theMerger,theErrorRescaling,theMatBeforeUpdate,theGeometry);
  }

  virtual void setHitCloner(TkCloner const * hc) {
  }


private:
  GsfPropagatorWithMaterial* thePropagator;
  const GsfPropagatorAdapter* theGeomPropagator;
  const FullConvolutionWithMaterial* theConvolutor;
  const TrajectoryStateUpdator* theUpdator;
  const MeasurementEstimator* theEstimator;
  const MultiTrajectoryStateMerger* theMerger;
  
  bool theTiming;
  bool theMatBeforeUpdate;
  float theErrorRescaling;

  const DetLayerGeometry dummyGeometry;
  const DetLayerGeometry* theGeometry;

};

#endif //TR_GsfTrajectorySmoother_H_
