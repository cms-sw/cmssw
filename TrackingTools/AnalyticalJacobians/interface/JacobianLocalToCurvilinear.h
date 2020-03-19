#ifndef JacobianLocalToCurvilinear_H
#define JacobianLocalToCurvilinear_H

#include "DataFormats/Math/interface/AlgebraicROOTObjects.h"
#include "DataFormats/GeometrySurface/interface/Surface.h"

#include "TrackingTools/TrajectoryParametrization/interface/LocalTrajectoryParameters.h"
#include "TrackingTools/TrajectoryParametrization/interface/GlobalTrajectoryParameters.h"

#include "FWCore/Utilities/interface/Visibility.h"

// class Surface;
class MagneticField;

/** Class which calculates the Jacobian matrix of the transformation
 *  from the local to the curvilinear frame. The Jacobian is calculated
 *  during construction and thereafter cached, enabling reuse of the same
 *  Jacobian without calculating it again.
 */

class JacobianLocalToCurvilinear {
public:
  /** Constructor from local trajectory parameters and surface defining the local frame. 
   *  NB!! No default constructor exists!
   */

  JacobianLocalToCurvilinear(const Surface& surface,
                             const LocalTrajectoryParameters& localParameters,
                             const MagneticField& magField);

  /** Constructor from local and global trajectory parameters and surface defining the local frame. 
   */

  JacobianLocalToCurvilinear(const Surface& surface,
                             const LocalTrajectoryParameters& localParameters,
                             const GlobalTrajectoryParameters& globalParameters,
                             const MagneticField& magField);

  /** Access to Jacobian.
   */

  const AlgebraicMatrix55& jacobian() const { return theJacobian; }

private:
  void compute(Surface::RotationType const& rot, LocalVector const& tnl, GlobalVector const& tn, GlobalVector const& hq)
      dso_internal;

  AlgebraicMatrix55 theJacobian;
};

#endif  //JacobianLocalToCurvilinear_H
