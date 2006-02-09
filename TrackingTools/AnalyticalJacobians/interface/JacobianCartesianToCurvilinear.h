#ifndef JacobianCartesianToCurvilinear_H
#define JacobianCartesianToCurvilinear_H

#include "Geometry/CommonDetAlgo/interface/AlgebraicObjects.h"

class GlobalTrajectoryParameters;

/** Class which calculates the Jacobian matrix of the transformation
 *  from the Cartesian to the curvilinear frame. The Jacobian is calculated
 *  during construction and thereafter cached, enabling reuse of the same
 *  Jacobian without calculating it again.
 */

class JacobianCartesianToCurvilinear {

 public:
  
  /** Constructor from global trajectory parameters. NB!! No default constructor exists!
   */
  
  JacobianCartesianToCurvilinear(const GlobalTrajectoryParameters& globalParameters);
  
  /** Access to Jacobian.
   */
  
  const AlgebraicMatrix& jacobian() const;


 private:
  
  AlgebraicMatrix theJacobian;

};  

#endif //JacobianCartesianToCurvilinear_H
