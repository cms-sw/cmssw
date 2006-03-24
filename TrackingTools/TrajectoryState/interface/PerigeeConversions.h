#ifndef PerigeeConversions_H
#define PerigeeConversions_H

#include "Geometry/CommonDetAlgo/interface/AlgebraicObjects.h"
#include "TrackingTools/TrajectoryState/interface/FreeTrajectoryState.h"
#include "TrackingTools/TrajectoryParametrization/interface/PerigeeTrajectoryParameters.h"
#include "MagneticField/Engine/interface/MagneticField.h"

class TrajectoryStateClosestToPoint;

/**
 * Class provides several methods to transform perigee parameters to and from
 * various other parametrisations.
 */

class PerigeeConversions
{
  typedef FreeTrajectoryState		FTS;

public:

   /**
   * This method calculates the perigee parameters from a given FTS
   * and a reference point.
   */

  PerigeeTrajectoryParameters ftsToPerigeeParameters(const FTS& originalFTS,
    const GlobalPoint& referencePoint) const;

  /**
   * This method returns the position (on the helix) at which the
   * parameters are defined
   */

  GlobalPoint positionFromPerigee(const PerigeeTrajectoryParameters& parameters,
    const GlobalPoint& referencePoint) const;

  /**
   * This method returns the (Cartesian) momentum.
   * The parameters need not be the full perigee parameters, as long as the first
   * 3 parameters are the transverse curvature, theta and phi.
   */

  GlobalVector momentumFromPerigee(const AlgebraicVector& momentum, 
    const TrackCharge& charge, const GlobalPoint& referencePoint,
    const MagneticField& magField)  const;

  /**
   * This method returns the (Cartesian) momentum from the PerigeeTrajectoryParameters
   */

  GlobalVector momentumFromPerigee (const PerigeeTrajectoryParameters& parameters,
    const GlobalPoint& referencePoint, const MagneticField& magField) const;

  /**
   * This method returns the charge.
   */

  TrackCharge chargeFromPerigee(const PerigeeTrajectoryParameters& perigee,
    const GlobalPoint& referencePoint) const;

  /**
   * Public constructor.
   * This constructor takes a momentum, with parameters
   * (transverse curvature, theta, phi) and a position, which is both the
   * reference position and the position at which the momentum is defined.
   * The covariance matrix is defined for these 6 parameters, in the order
   * (x, y, z, transverse curvature, theta, phi).
   */
  TrajectoryStateClosestToPoint trajectoryStateClosestToPoint
	(const AlgebraicVector& momentum, const GlobalPoint& referencePoint,
	 const TrackCharge& charge, const AlgebraicMatrix& theCovarianceMatrix,
	 const MagneticField& magField) const;


private:

  /**
   * Jacobians of tranformations between the parametrixation
   * (x, y, z, transverse curvature, theta, phi) to Cartesian
   */

  AlgebraicMatrix  jacobianParameters2Cartesian
	(const AlgebraicVector& momentum, const GlobalPoint& position,
	 const TrackCharge& charge, const MagneticField& magField) const;

};
#endif
