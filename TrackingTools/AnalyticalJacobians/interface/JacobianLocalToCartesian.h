#ifndef JacobianLocalToCartesian_H
#define JacobianLocalToCartesian_H

#include "DataFormats/CLHEP/interface/AlgebraicObjects.h"

class Surface;
class LocalTrajectoryParameters;

/** Class which calculates the Jacobian matrix of the transformation
 *  from the local to the Caresian frame. The Jacobian is calculated
 *  during construction and thereafter cached, enabling reuse of the same
 *  Jacobian without calculating it again.
 */

class JacobianLocalToCartesian {

 public:

  /** Constructor from local trajectory parameters and surface defining the local frame. 
   *  NB!! No default constructor exists!
   */
  
  JacobianLocalToCartesian(const Surface& surface, const LocalTrajectoryParameters& localParameters);
  
  /** Access to Jacobian.
   */
  
  const AlgebraicMatrix65& jacobian() const;
  const AlgebraicMatrix jacobian_old() const;


 private:
  
  AlgebraicMatrix65 theJacobian;

};  

#endif //JacobianLocalToCartesian_H
