#ifndef JacobianCurvilinearToLocal_H
#define JacobianCurvilinearToLocal_H

#include "DataFormats/Math/interface/AlgebraicROOTObjects.h"
#include "DataFormats/GeometrySurface/interface/Surface.h"

#include "FWCore/Utilities/interface/Visibility.h"


class LocalTrajectoryParameters;
class GlobalTrajectoryParameters;
class MagneticField;
class GlobalVector;

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
  
  JacobianCurvilinearToLocal(const Surface& surface, 
			     const GlobalTrajectoryParameters& globalParameters,
			     const MagneticField& magField) : theJacobian() 

  /** Access to Jacobian.
   */
  
  const AlgebraicMatrix55& jacobian() const { return  theJacobian; }


 private:

  void compute(Surface::RotationType const & rot, GlobalVector  const & tn, GlobalVector const & qh)  dso_internal;  

  AlgebraicMatrix55 theJacobian;

};  

#endif //JacobianCurvilinearToLocal_H
