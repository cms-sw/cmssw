#ifndef TrackAssociator_MuonSegmentMatch_h
#define TrackAssociator_MuonSegmentMatch_h
#include "DataFormats/Math/interface/Vector3D.h"
#include "DataFormats/Math/interface/Point3D.h"

class MuonSegmentMatch {
 public:
   math::XYZPoint  segmentGlobalPosition;
   math::XYZPoint  segmentLocalPosition;
   math::XYZVector segmentLocalDirection;
   float  segmentLocalErrorXX;
   float  segmentLocalErrorYY;
   float  segmentLocalErrorXY;
   float  segmentLocalErrorDxDz;
   float  segmentLocalErrorDyDz;

   math::XYZPoint  trajectoryGlobalPosition;
   math::XYZPoint  trajectoryLocalPosition;
   math::XYZVector trajectoryLocalDirection;
   float  trajectoryLocalErrorXX;
   float  trajectoryLocalErrorYY;
   float  trajectoryLocalErrorXY;
   float  trajectoryLocalErrorDxDz;
   float  trajectoryLocalErrorDyDz;

   unsigned int id;
};
#endif
