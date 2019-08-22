#ifndef JacobianCartesianToLocal_H
#define JacobianCartesianToLocal_H

#include "DataFormats/Math/interface/AlgebraicROOTObjects.h"

class Surface;
class LocalTrajectoryParameters;

/** Class which calculates the Jacobian matrix of the transformation
 *  from the Cartesian to the local frame. The Jacobian is calculated
 *  during construction and thereafter cached, enabling reuse of the same
 *  Jacobian without calculating it again.
 */

class JacobianCartesianToLocal {
public:
  /** Constructor from local trajectory parameters and surface defining the local frame. 
   *  NB!! No default constructor exists!
   */

  JacobianCartesianToLocal(const Surface& surface, const LocalTrajectoryParameters& localParameters);

  /** Access to Jacobian.
   */

  const AlgebraicMatrix56& jacobian() const { return theJacobian; }

private:
  AlgebraicMatrix56 theJacobian;
};

#endif  //JacobianCartesianToLocal_H
