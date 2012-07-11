#ifndef JacobianCurvilinearToLocal_H
#define JacobianCurvilinearToLocal_H

#include "DataFormats/CLHEP/interface/AlgebraicObjects.h"

class Surface;
class LocalTrajectoryParameters;
class MagneticField;

/** Class which calculates the Jacobian matrix of the transformation
 *  from the curvilinear to the local frame. The Jacobian is calculated
 *  during construction and thereafter cached, enabling reuse of the same
 *  Jacobian without calculating it again.
 */

class JacobianCurvilinearToLocal {

 public:

  /** Constructor from local trajectory parameters and surface defining the local frame. 
   *  NB!! No default constructor exists!
   */
  
  JacobianCurvilinearToLocal(const Surface& surface, 
			     const LocalTrajectoryParameters& localParameters,
			     const MagneticField& magField);
  
  /** Access to Jacobian.
   */
  
  const AlgebraicMatrix55& jacobian() const;
  const AlgebraicMatrix jacobian_old() const;


 private:
  
  AlgebraicMatrix55 theJacobian;

};  

#endif //JacobianCurvilinearToLocal_H
