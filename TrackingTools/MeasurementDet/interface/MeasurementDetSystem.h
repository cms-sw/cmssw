#ifndef MeasurementDetSystem_H
#define MeasurementDetSystem_H

#include "DataFormats/DetId/interface/DetId.h"
class MeasurementDet;

class MeasurementDetSystem {
public:

  /// Return the pointer to the MeasurementDet corresponding to a given DetId
  virtual const MeasurementDet*       idToDet(const DetId& id) const = 0;


};

#endif
