#ifndef TrackAssociator_TAMuonSegmentMatch_h
#define TrackAssociator_TAMuonSegmentMatch_h
#include "DataFormats/Math/interface/Vector3D.h"
#include "DataFormats/Math/interface/Point3D.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/CSCRecHit/interface/CSCSegmentCollection.h"
#include "DataFormats/DTRecHit/interface/DTRecSegment4DCollection.h"

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
   DTRecSegment4DRef  dtSegmentRef;
   CSCSegmentRef      cscSegmentRef;
};
#endif
