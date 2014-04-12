#ifndef CurvilinearJacobian_H
#define CurvilinearJacobian_H

#include "DataFormats/Math/interface/AlgebraicROOTObjects.h"

/** Base class for calculations of Jacobians of transformations within the curvilinear frame.
 */

class CurvilinearJacobian {
public:

  CurvilinearJacobian()  : theJacobian(AlgebraicMatrixID()){}

  virtual ~CurvilinearJacobian() {}

  const AlgebraicMatrix55& jacobian() const {return theJacobian;}

protected:
  
  AlgebraicMatrix55 theJacobian;


};  

#endif
