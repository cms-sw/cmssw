#ifndef Tracker_MeasurementEstimator_H
#define Tracker_MeasurementEstimator_H

#include "DataFormats/GeometryVector/interface/Vector2DBase.h"
#include "DataFormats/GeometryVector/interface/LocalTag.h"
#include <utility>

class BoundPlane;
class TrajectoryStateOnSurface;
class Surface;
class TransientTrackingRecHit;

/** The MeasurementEstimator defines the compatibility of a 
 *  TrajectoryStateOnSurface and a RecHit, and of a 
 *  TrajectoryStateOnSurface and a BoundPlane.
 *  It is used in the Det interface to obtain compatible measurements.
 */


class MeasurementEstimator {
public:

  typedef Vector2DBase< float, LocalTag>    Local2DVector;

  virtual ~MeasurementEstimator() {}

  typedef std::pair<bool,double>     HitReturnType;
  typedef bool                  SurfaceReturnType;

  /** Returns pair( true, value) if the TrajectoryStateOnSurface is compatible
   *  with the RecHit, and pair( false, value) if it is not compatible.
   *  The TrajectoryStateOnSurface must be on the same Surface as the RecHit. 
   *  For an estimator where there is no value computed, e.g. fixed
   *  window estimator, only the first(bool) part is of interest.
   */
  virtual HitReturnType estimate( const TrajectoryStateOnSurface& ts, 
				  const TransientTrackingRecHit& hit) const = 0;

  /** Returns true if the TrajectoryStateOnSurface is compatible with the
   *  BoundPlane, false otherwise.
   *  The TrajectoryStateOnSurface must be on the plane.
   */
  virtual SurfaceReturnType estimate( const TrajectoryStateOnSurface& ts, 
				      const BoundPlane& plane) const = 0;

/*   virtual SurfaceReturnType estimate( const TrajectoryStateOnSurface& ts,  */
/* 				      const BoundSurface& plane) const; */

  virtual MeasurementEstimator* clone() const = 0;

  /** Returns the size of the compatibility region around the local position of the 
   *  TrajectoryStateOnSurface along the directions of local x and y axis.
   *  The TrajectoryStateOnSurface must be on the plane.
   *  This method allows to limit the search for compatible detectors or RecHits.
   *  The MeasurementEstimator should not return "true" for any RecHit or
   *  BoundPlane which is entirely outside of the compatibility region defined 
   *  by maximalLocalDisplacement().
   */
  virtual Local2DVector 
  maximalLocalDisplacement( const TrajectoryStateOnSurface& ts,
			    const BoundPlane& plane) const;
};

#endif // Tracker_MeasurementEstimator_H
