#ifndef JacobianCurvilinearToCartesian_H
#define JacobianCurvilinearToCartesian_H

#include "DataFormats/CLHEP/interface/AlgebraicObjects.h"

class GlobalTrajectoryParameters;

/** Class which calculates the Jacobian matrix of the transformation
 *  from the curvilinear to the Cartesian frame. The Jacobian is calculated
 *  during construction and thereafter cached, enabling reuse of the same
 *  Jacobian without calculating it again.
 */

class JacobianCurvilinearToCartesian {

 public:
  
  /** Constructor from global trajectory parameters. NB!! No default constructor exists!
   */
  
  JacobianCurvilinearToCartesian(const GlobalTrajectoryParameters& globalParameters);
  
  /** Access to Jacobian.
   */
  
  const AlgebraicMatrix65& jacobian() const;
  const AlgebraicMatrix jacobian_old() const;


 private:
  
  AlgebraicMatrix65 theJacobian;

};  

#endif //JacobianCurvilinearToCartesian_H
