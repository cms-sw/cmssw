#ifndef MeasurementDetSystem_H
#define MeasurementDetSystem_H

#include "DataFormats/DetId/interface/DetId.h"
#include "TrackingTools/MeasurementDet/interface/MeasurementDetWithData.h" 


class MeasurementDetSystem {
public:
  virtual ~MeasurementDetSystem() = default;
  /// Return the pointer to the MeasurementDet corresponding to a given DetId
  /// needs the data, as it could do on-demand unpacking or similar things
  virtual MeasurementDetWithData idToDet(const DetId& id, const MeasurementTrackerEvent &data) const = 0;
};

#endif
