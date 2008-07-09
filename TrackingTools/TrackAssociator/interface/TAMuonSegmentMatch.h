#ifndef TrackAssociator_TAMuonSegmentMatch_h
#define TrackAssociator_TAMuonSegmentMatch_h
#include "DataFormats/Math/interface/Vector3D.h"
#include "DataFormats/Math/interface/Point3D.h"
#include "DataFormats/DetId/interface/DetId.h"

class TAMuonSegmentMatch {
 public:
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
   float  t0;
   bool   hasZed;
   bool   hasPhi;
};
#endif
