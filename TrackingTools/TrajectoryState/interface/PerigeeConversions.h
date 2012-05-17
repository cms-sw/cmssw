#ifndef PerigeeConversions_H
#define PerigeeConversions_H

#include "DataFormats/CLHEP/interface/AlgebraicObjects.h"
#include "TrackingTools/TrajectoryState/interface/FreeTrajectoryState.h"
#include "TrackingTools/TrajectoryParametrization/interface/PerigeeTrajectoryParameters.h"
#include "TrackingTools/TrajectoryParametrization/interface/PerigeeTrajectoryError.h"

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
    const GlobalPoint& referencePoint, double& pt) const;

  PerigeeTrajectoryError ftsToPerigeeError(const FTS& originalFTS) const;

//   PerigeeTrajectoryParameters helixToPerigeeParameters
//     (const reco::helix::Parameters & helixPar, const GlobalPoint& referencePoint) const;
// 
//   PerigeeTrajectoryError helixToPerigeeError(const reco::helix::Parameters & helixPar, 
// 	const reco::helix::Covariance & helixCov) const;



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
    const MagneticField* field)  const;

  GlobalVector momentumFromPerigee(const AlgebraicVector3& momentum, 
    const TrackCharge& charge, const GlobalPoint& referencePoint,
    const MagneticField* field)  const;

  /**
   * This method returns the (Cartesian) momentum from the PerigeeTrajectoryParameters
   */

  GlobalVector momentumFromPerigee (const PerigeeTrajectoryParameters& parameters,
				    double pt,
				    const GlobalPoint& referencePoint) const;

  /**
   * This method returns the charge.
   */

  TrackCharge chargeFromPerigee(const PerigeeTrajectoryParameters& perigee) const;

  CurvilinearTrajectoryError curvilinearError(const PerigeeTrajectoryError& perigeeError,
    const GlobalTrajectoryParameters& gtp) const;


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
	 const MagneticField* field) const;

  TrajectoryStateClosestToPoint trajectoryStateClosestToPoint
	(const AlgebraicVector3& momentum, const GlobalPoint& referencePoint,
	 const TrackCharge& charge, const AlgebraicSymMatrix66& theCovarianceMatrix,
	 const MagneticField* field) const;


  /**
   * Jacobians of tranformations between the parametrixation
   * (x, y, z, transverse curvature, theta, phi) to Cartesian
   */

  AlgebraicMatrix  jacobianParameters2Cartesian_old
	(const AlgebraicVector& momentum, const GlobalPoint& position,
	 const TrackCharge& charge, const MagneticField* field) const;
 
  /**
   * Jacobians of tranformations between the parametrixation
   * (x, y, z, transverse curvature, theta, phi) to Cartesian
   */

  AlgebraicMatrix66  jacobianParameters2Cartesian
	(const AlgebraicVector3& momentum, const GlobalPoint& position,
	 const TrackCharge& charge, const MagneticField* field) const;




  /**
   * Jacobians of tranformations between curvilinear frame at point of closest
   * approach in transverse plane and perigee frame. The fts must therefore be
   * given at exactly this point in order to yield the correct Jacobians.
   */

  AlgebraicMatrix jacobianCurvilinear2Perigee_old(const FreeTrajectoryState& fts) const;

  AlgebraicMatrix jacobianPerigee2Curvilinear_old(const GlobalTrajectoryParameters& gtp) const;

  AlgebraicMatrix55 jacobianCurvilinear2Perigee(const FreeTrajectoryState& fts) const;

  AlgebraicMatrix55 jacobianPerigee2Curvilinear(const GlobalTrajectoryParameters& gtp) const;


//   AlgebraicMatrix jacobianHelix2Perigee(const reco::helix::Parameters & helixPar, 
// 	const reco::helix::Covariance & helixCov) const;


};
#endif
