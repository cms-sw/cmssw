#ifndef JacobianLocalToCurvilinear_H
#define JacobianLocalToCurvilinear_H

#include "DataFormats/CLHEP/interface/AlgebraicObjects.h"

class Surface;
class LocalTrajectoryParameters;
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
  
  /** Access to Jacobian.
   */
  
  const AlgebraicMatrix55& jacobian() const;
  const AlgebraicMatrix jacobian_old() const;


 private:
  
  AlgebraicMatrix55 theJacobian;

};  

#endif //JacobianLocalToCurvilinear_H
