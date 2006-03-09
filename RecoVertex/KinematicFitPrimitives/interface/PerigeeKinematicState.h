#ifndef PerigeeKinematicState_H
#define PerigeeKinematicState_H

#include "RecoVertex/KinematicFitPrimitives/interface/KinematicState.h"
#include "RecoVertex/KinematicFitPrimitives/interface/ExtendedPerigeeTrajectoryParameters.h"
#include "RecoVertex/KinematicFitPrimitives/interface/ExtendedPerigeeTrajectoryError.h"

/**
 * Class caching the "extended"
 * perigee parametrization for
 * vertex fitting inside the 
 * KinematicFit library.
 * Extended parameters are:
 * (epsilon, rho, phi, theta_p, z_p, m)
 * (see TrajectoryStateClosestToPoint
 * class for reference)
 *
 * Kirill Prokofiev, august 2003
 */

class PerigeeKinematicState
{
public:

 PerigeeKinematicState()
 {
  vl = false;
  errorIsAvailable = false;
 }

 virtual ~PerigeeKinematicState(){}
/**
 * Access methods
 */
 bool hasError() const
 {
  if(!(isValid()))throw VertexException("PerigeeKinematicState::error is requested for the invalid state");
  return errorIsAvailable;
 }
 
 bool isValid() const
 {return vl;}

 const KinematicState theState() const
 {
  if(!isValid()) throw VertexException("PerigeeKinematicState::initial state is requested for the invalid state");
  return inState;
 }

/**
 * Returns the reference point
 */
 const GlobalPoint referencePoint() const
 {
  if(!isValid()) throw VertexException("PerigeeKinematicState::point is requested for the invalid state");
  return point;
 }

/**
 * Returns the error matrix of extended
 * perigee parametrization
 */    
 const ExtendedPerigeeTrajectoryError& perigeeError() const
 {
  if(!(isValid()))  throw VertexException("PerigeeKinematicState::requesting perigee error for invalid state");
  if(!(hasError())) throw VertexException("PerigeeKinematicState::requesting perigee error when none available");
  return cov;
 }

/**
 * Returns the extended perigee parameters
 */
 const ExtendedPerigeeTrajectoryParameters & perigeeParameters() const
 {
  if(!(isValid()))  throw VertexException("PerigeeKinematicState::requesting perigee parameters for invalid state");
  return par;
 }

private:

 friend class TransientTrackKinematicStateBuilder;


 PerigeeKinematicState(const KinematicState& state, const GlobalPoint& pt);
/*
 AlgebraicMatrix jacobianKinematicToExPerigee(const KinematicState& state, 
                                              const GlobalPoint& pt)const;
 AlgebraicMatrix jacobianExPerigeeToKinematic(const ExtendedPerigeeTrajectoryParameters& state,
                                              const GlobalPoint& point)const;
*/					      
 AlgebraicMatrix jacobianCurvilinear2Perigee(const FreeTrajectoryState& fts) const;
 
private:
 GlobalPoint point;
 ExtendedPerigeeTrajectoryParameters par;
 ExtendedPerigeeTrajectoryError cov;
 KinematicState inState;
 bool errorIsAvailable;
 bool vl;

};
#endif
