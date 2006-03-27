#ifndef TrackingSystem_H
#define TrackingSystem_H

class DetId;
class MeasurementDet;

class TrackingSystem {
public:

  virtual const MeasurementDet* measurementDet(const DetId& id) const = 0;
 

};

#endif
