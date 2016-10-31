#ifndef Tracker_MeasurementEstimator_H
#define Tracker_MeasurementEstimator_H

#include "DataFormats/GeometryVector/interface/Vector2DBase.h"
#include "DataFormats/GeometryVector/interface/LocalTag.h"
#include<limits>

class Plane;
class TrajectoryStateOnSurface;
class Surface;
class TrackingRecHit;

/** The MeasurementEstimator defines the compatibility of a 
 *  TrajectoryStateOnSurface and a RecHit, and of a 
 *  TrajectoryStateOnSurface and a Plane.
 *  It is used in the Det interface to obtain compatible measurements.
 */


class MeasurementEstimator {
public:

  struct OpaquePayload { virtual ~OpaquePayload(){} int tag=0;};

  using Local2DVector = Vector2DBase< float, LocalTag>;


  MeasurementEstimator() {}
  MeasurementEstimator(float maxSag, float minToll, float mpt) :
     m_maxSagitta(maxSag),
     m_minTolerance2(minToll*minToll),
     m_minPt2ForHitRecoveryInGluedDet(mpt*mpt)
     {}

  virtual ~MeasurementEstimator() {}

  using HitReturnType     = std::pair<bool,double>;
  using SurfaceReturnType = bool;

  /** Returns pair( true, value) if the TrajectoryStateOnSurface is compatible
   *  with the RecHit, and pair( false, value) if it is not compatible.
   *  The TrajectoryStateOnSurface must be on the same Surface as the RecHit. 
   *  For an estimator where there is no value computed, e.g. fixed
   *  window estimator, only the first(bool) part is of interest.
   */
  virtual HitReturnType estimate( const TrajectoryStateOnSurface& ts, 
				  const TrackingRecHit& hit) const = 0;

  /* verify the compatibility of the Hit with the Trajectory based
   * on hit properties other than those used in estimate 
   * (that usually computes the compatibility of the Trajectory with the Hit)
   * 
   */
  virtual bool preFilter(const TrajectoryStateOnSurface&, OpaquePayload const &) const { return true;}


  /** Returns true if the TrajectoryStateOnSurface is compatible with the
   *  Plane, false otherwise.
   *  The TrajectoryStateOnSurface must be on the plane.
   */
  virtual SurfaceReturnType estimate( const TrajectoryStateOnSurface& ts, 
				      const Plane& plane) const = 0;

  virtual MeasurementEstimator* clone() const = 0;

  /** Returns the size of the compatibility region around the local position of the 
   *  TrajectoryStateOnSurface along the directions of local x and y axis.
   *  The TrajectoryStateOnSurface must be on the plane.
   *  This method allows to limit the search for compatible detectors or RecHits.
   *  The MeasurementEstimator should not return "true" for any RecHit or
   *  Plane which is entirely outside of the compatibility region defined 
   *  by maximalLocalDisplacement().
   */
  virtual Local2DVector 
  maximalLocalDisplacement( const TrajectoryStateOnSurface& ts,
			    const Plane& plane) const=0;

  float maxSagitta() const { return m_maxSagitta;}
  float	minTolerance2() const { return m_minTolerance2;}
  float	minPt2ForHitRecoveryInGluedDet() const { return m_minPt2ForHitRecoveryInGluedDet;}

private:
  /*
   *  why here? 
   * MeasurementEstimator is the only configurable item that percolates down to geometry event by event (actually hit by hit) and not at initialization time
   * It is therefore the natural candidate to collect all parameters that affect pattern-recongnition 
   * and require to be controlled with higher granularity than job level (such as iteration by iteration)
   */ 
  float m_maxSagitta=-1.; // maximal sagitta for linear approximation
  float m_minTolerance2=100.; // square of minimum tolerance ot be considered inside a detector
  float m_minPt2ForHitRecoveryInGluedDet=std::numeric_limits<float>::max();
};

#endif // Tracker_MeasurementEstimator_H
