#ifndef TrackAssociator_MuonSegmentMatch_h
#define TrackAssociator_MuonSegmentMatch_h
#include "DataFormats/Math/interface/Vector3D.h"
#include "DataFormats/Math/interface/Point3D.h"
#include "DataFormats/DetId/interface/DetId.h"

class MuonSegmentMatch {
 public:
   double pullX() const;
   double pullY() const;

   math::XYZPoint  segmentGlobalPosition;
   math::XYZPoint  segmentLocalPosition;
   math::XYZVector segmentLocalDirection;
   float  segmentLocalErrorXX;
   float  segmentLocalErrorYY;
   float  segmentLocalErrorXY;
   float  segmentLocalErrorDxDz;
   float  segmentLocalErrorDyDz;
   float  segmentLocalErrorXDxDz;
   float  segmentLocalErrorYDyDz;

   math::XYZPoint  trajectoryGlobalPosition;
   math::XYZPoint  trajectoryLocalPosition;
   math::XYZVector trajectoryLocalDirection;
   float  trajectoryLocalErrorXX;
   float  trajectoryLocalErrorYY;
   float  trajectoryLocalErrorXY;
   float  trajectoryLocalErrorDxDz;
   float  trajectoryLocalErrorDyDz;
   float  trajectoryLocalErrorXDxDz;
   float  trajectoryLocalErrorYDyDz;
};
#endif
