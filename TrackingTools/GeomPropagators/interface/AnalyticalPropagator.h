#ifndef _COMMONRECO_ANALYTICALPROPAGATOR_H_
#define _COMMONRECO_ANALYTICALPROPAGATOR_H_

#include "TrackingTools/GeomPropagators/interface/Propagator.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
#include "FWCore/Utilities/interface/Visibility.h"

#include <cmath>
#include <cfloat>

class Surface;
class Cylinder;
class Plane;
class HelixPlaneCrossing;
class MagneticField; 


/** (Mostly) analytical helix propagation to cylindrical or planar surfaces.
 *  Based on GtfGeometricalPropagator with successive replacement of components
 *  (currently: propagation to arbitrary plane).
 */

class AnalyticalPropagator GCC11_FINAL : public Propagator {

public:

  AnalyticalPropagator( const MagneticField* field,
		        PropagationDirection dir = alongMomentum,
			float maxDPhi = 1.6,bool isOld=true) :
    Propagator(dir),
    theMaxDPhi2(maxDPhi*maxDPhi),
    theMaxDBzRatio(0.5),
    theField(field),
    isOldPropagationType(isOld) {}

  ~AnalyticalPropagator() {}
  //
  // use base class methods where necessary:
  // - propagation from TrajectoryStateOnSurface 
  //     (will use propagation from FreeTrajectoryState)
  // - propagation to general Surface
  //     (will use specialised methods for planes or cylinders)
  //
  using Propagator::propagate;
  using Propagator::propagateWithPath;

  /// propagation to plane
  TrajectoryStateOnSurface propagate(const FreeTrajectoryState& fts, 
                                     const Plane& plane) const {
    return propagateWithPath(fts,plane).first;
  }
  /// propagation to plane with path length  
  std::pair<TrajectoryStateOnSurface,double> 
  propagateWithPath(const FreeTrajectoryState& fts, 
		    const Plane& plane) const; 
  
  /// propagation to cylinder
  TrajectoryStateOnSurface propagate(const FreeTrajectoryState& fts, 
                                     const Cylinder& cylinder) const {
    return propagateWithPath(fts,cylinder).first;
  }
  /// propagation to cylinder with path length
  std::pair<TrajectoryStateOnSurface,double> 
  propagateWithPath(const FreeTrajectoryState& fts, 
		    const Cylinder& cylinder) const;
  /** limitation of change in transverse direction
   *  (to avoid loops).
   */
  virtual bool setMaxDirectionChange( float phiMax) { 
    theMaxDPhi2 = phiMax*phiMax;
    return true;
  }
  
#ifndef CMS_NO_RELAXED_RETURN_TYPE
  virtual AnalyticalPropagator * clone() const 
#else
    virtual Propagator * clone() const
#endif
  {
    return new AnalyticalPropagator(*this);
  }
  
  /** Set the maximum relative change in Bz (Bz_at_end-Bz_at_start)/Bz_at_start
   * for a single propagation. The default is no limit.
   * NB: this propagator assumes constant, non-zero magnetic field parallel to the z-axis!
   **/
  void setMaxRelativeChangeInBz (const float maxDBz) {
    theMaxDBzRatio = maxDBz;
  }


private:
  /// propagation of errors (if needed) and generation of a new TSOS
  std::pair<TrajectoryStateOnSurface,double> 
  propagatedStateWithPath (const FreeTrajectoryState& fts, 
			   const Surface& surface, 
			   const GlobalTrajectoryParameters& gtp, 
			   const double& s) const dso_internal;

  /// parameter propagation to cylinder (returns position, momentum and path length)
  bool propagateParametersOnCylinder(const FreeTrajectoryState& fts, 
				     const Cylinder& cylinder, 
				     GlobalPoint& x, 
				     GlobalVector& p, 
				     double& s) const dso_internal;

  /// parameter propagation to plane (returns position, momentum and path length)
  bool propagateParametersOnPlane(const FreeTrajectoryState& fts, 
				  const Plane& plane, 
				  GlobalPoint& x, 
				  GlobalVector& p, 
				  double& s) const dso_internal;
  
  /// straight line parameter propagation to a plane
  bool propagateWithLineCrossing(const GlobalPoint&, const GlobalVector&, 
				 const Plane&, GlobalPoint&, double&) const dso_internal;
  /// straight line parameter propagation to a cylinder
  bool propagateWithLineCrossing(const GlobalPoint&, const GlobalVector&, 
				 const Cylinder&, GlobalPoint&, double&) const dso_internal;
  /// helix parameter propagation to a plane using HelixPlaneCrossing
  bool propagateWithHelixCrossing(HelixPlaneCrossing&, const Plane&, const float,
				  GlobalPoint&, GlobalVector&, double& s) const dso_internal;

  virtual const MagneticField* magneticField() const {return theField;}

private:
  typedef std::pair<TrajectoryStateOnSurface,double> TsosWP;
  float theMaxDPhi2;
  float theMaxDBzRatio;
  const MagneticField* theField;
  bool isOldPropagationType;
};

#endif
