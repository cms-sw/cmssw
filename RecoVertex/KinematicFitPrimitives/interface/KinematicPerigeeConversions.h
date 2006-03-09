#ifndef KinematicPerigeeConversions_H
#define KinematicPerigeeConversions_h

#include "RecoVertex/KinematicFitPrimitives/interface/KinematicParameters.h"
#include "RecoVertex/KinematicFitPrimitives/interface/KinematicState.h"
#include "RecoVertex/KinematicFitPrimitives/interface/ExtendedPerigeeTrajectoryParameters.h"

/**
 * Helper class to simplify parameters 
 * conversions between kinematic
 * and extended perigee parametrization
 *
 * Kirill Prokofiev, August 2003
 */
class KinematicPerigeeConversions
{
public:

 KinematicPerigeeConversions()
 {}
 
 ExtendedPerigeeTrajectoryParameters extendedPerigeeFromKinematicParameters(const KinematicState& state,
                                                                            const GlobalPoint& point) const;
					   
 KinematicParameters kinematicParametersFromExPerigee(const ExtendedPerigeeTrajectoryParameters& pr, 
                                                      const GlobalPoint& point )const;
						      
 KinematicState kinematicState(const AlgebraicVector& momentum, const GlobalPoint& referencePoint,
	                         const TrackCharge& charge, const AlgebraicMatrix& theCovarianceMatrix) const;
				 
 AlgebraicVector momentumFromPerigee(const AlgebraicVector& momentum,
                                     const GlobalPoint& referencePoint, const TrackCharge& ch)const;
				     
private:
  /**
   * Jacobians of tranformations from the parametrixation
   * (x, y, z, transverse curvature, theta, phi,m) to kinematic
   *  parameters
   */
AlgebraicMatrix jacobianParameters2Kinematic(const AlgebraicVector& momentum, 
                                                           const GlobalPoint& referencePoint,
	                                                   const TrackCharge& charge)const;
};
#endif
