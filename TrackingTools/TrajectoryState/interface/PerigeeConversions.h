#ifndef PerigeeConversions_H
#define PerigeeConversions_H

#include "DataFormats/Math/interface/AlgebraicROOTObjects.h"
#include "TrackingTools/TrajectoryState/interface/FreeTrajectoryState.h"
#include "TrackingTools/TrajectoryParametrization/interface/PerigeeTrajectoryParameters.h"
#include "TrackingTools/TrajectoryParametrization/interface/PerigeeTrajectoryError.h"

class TrajectoryStateClosestToPoint;

/**
 * namespace provides several functions to transform perigee parameters to and from
 * various other parametrisations.
 */
namespace PerigeeConversions {
  typedef FreeTrajectoryState		FTS;
  /**
   *  calculates the perigee parameters from a given FTS
   * and a reference point.
   */
  PerigeeTrajectoryParameters ftsToPerigeeParameters(const FTS& originalFTS,
    const GlobalPoint& referencePoint, double& pt);

  PerigeeTrajectoryError ftsToPerigeeError(const FTS& originalFTS);


  /**
   *  returns the position (on the helix) at which the
   * parameters are defined
   */
  GlobalPoint positionFromPerigee(const PerigeeTrajectoryParameters& parameters,
    const GlobalPoint& referencePoint);

  /**
   *  returns the (Cartesian) momentum.
   * The parameters need not be the full perigee parameters, as long as the first
   * 3 parameters are the transverse curvature, theta and phi.
   */
   GlobalVector momentumFromPerigee(const AlgebraicVector3& momentum, 
    const TrackCharge& charge, const GlobalPoint& referencePoint,
    const MagneticField* field);

  /**
   *  returns the (Cartesian) momentum from the PerigeeTrajectoryParameters
   */
  GlobalVector momentumFromPerigee (const PerigeeTrajectoryParameters& parameters,
				    double pt,
				    const GlobalPoint& referencePoint);


  CurvilinearTrajectoryError curvilinearError(const PerigeeTrajectoryError& perigeeError,
    const GlobalTrajectoryParameters& gtp);


  /**
   * Public constructor.
   * This constructor takes a momentum, with parameters
   * (transverse curvature, theta, phi) and a position, which is both the
   * reference position and the position at which the momentum is defined.
   * The covariance matrix is defined for these 6 parameters, in the order
   * (x, y, z, transverse curvature, theta, phi).
   */
  TrajectoryStateClosestToPoint trajectoryStateClosestToPoint
	(const AlgebraicVector3& momentum, const GlobalPoint& referencePoint,
	 const TrackCharge& charge, const AlgebraicSymMatrix66& theCovarianceMatrix,
	 const MagneticField* field);


/**
   * Jacobians of tranformations between the parametrixation
   * (x, y, z, transverse curvature, theta, phi) to Cartesian
   */
  AlgebraicMatrix66  jacobianParameters2Cartesian
	(const AlgebraicVector3& momentum, const GlobalPoint& position,
	 const TrackCharge& charge, const MagneticField* field);


  /**
   * Jacobians of tranformations between curvilinear frame at point of closest
   * approach in transverse plane and perigee frame. The fts must therefore be
   * given at exactly this point in order to yield the correct Jacobians.
   */
  AlgebraicMatrix55 jacobianCurvilinear2Perigee(const FreeTrajectoryState& fts);

  AlgebraicMatrix55 jacobianPerigee2Curvilinear(const GlobalTrajectoryParameters& gtp);


}

#endif
