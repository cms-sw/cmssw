#ifndef TrackAssociator_MuonSegmentMatch_h
#define TrackAssociator_MuonSegmentMatch_h
#include "DataFormats/Math/interface/Vector3D.h"
#include "DataFormats/Math/interface/Point3D.h"
#include "DataFormats/DetId/interface/DetId.h"

class MuonSegmentMatch {
 public:
   int station() const;
   int detector() const { return id.subdetId(); }
   double pullX(bool scaled = true) const;
   double pullY(bool scaled = true) const;
   double scaledSegmentLocalErrorXX() const;
   double scaledSegmentLocalErrorYY() const;
   double scaledTrajectoryLocalErrorXX() const;
   double scaledTrajectoryLocalErrorYY() const;

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

   DetId id;
};
#endif
