#ifndef CaloSimAlgos_CaloShapes_h
#define CaloSimAlgos_CaloShapes_h

/** A mechanism to let you pick shapes, based on DetId
*/
#include "DataFormats/DetId/interface/DetId.h"
class CaloVShape;

class CaloShapes
{
public:
  CaloShapes(): theShape(0) {}
  // doesn't take ownership of the pointer
  CaloShapes(const CaloVShape * shape) : theShape(shape) {}
  virtual const CaloVShape * shape(const DetId & detId, bool precise=false) const {return theShape;}
  virtual ~CaloShapes() = default;
private:
  const CaloVShape * theShape;
};

#endif

