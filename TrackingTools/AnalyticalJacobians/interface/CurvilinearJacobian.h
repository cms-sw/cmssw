#ifndef CurvilinearJacobian_H
#define CurvilinearJacobian_H

#include "Geometry/CommonDetAlgo/interface/AlgebraicObjects.h"
#include "Geometry/Vector/interface/GlobalPoint.h"
#include "Geometry/Vector/interface/GlobalVector.h"


/** Base class for calculations of Jacobians of transformations within the curvilinear frame.
 */

class CurvilinearJacobian {
public:

  CurvilinearJacobian() {}

  virtual ~CurvilinearJacobian() {}

  virtual const AlgebraicMatrix& jacobian() const = 0;

};  

#endif
