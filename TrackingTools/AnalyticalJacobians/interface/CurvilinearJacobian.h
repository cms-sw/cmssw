#ifndef CurvilinearJacobian_H
#define CurvilinearJacobian_H

#include "DataFormats/CLHEP/interface/AlgebraicObjects.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "DataFormats/GeometryVector/interface/GlobalVector.h"

/** Base class for calculations of Jacobians of transformations within the curvilinear frame.
 */

class CurvilinearJacobian {
public:

  CurvilinearJacobian() {}

  virtual ~CurvilinearJacobian() {}

  virtual const AlgebraicMatrix55& jacobian() const = 0;
  virtual const AlgebraicMatrix jacobian_old() const = 0;

};  

#endif
