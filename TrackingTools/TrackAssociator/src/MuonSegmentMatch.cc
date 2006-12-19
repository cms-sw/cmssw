#include "TrackingTools/TrackAssociator/interface/MuonSegmentMatch.h"
#include "DataFormats/MuonDetId/interface/DTChamberId.h"
#include "DataFormats/MuonDetId/interface/CSCDetId.h"
#include "DataFormats/MuonDetId/interface/RPCDetId.h"

double MuonSegmentMatch::pullX() const {
   double deltaX, errorX;
   deltaX = segmentLocalPosition.X()-trajectoryLocalPosition.X();
   errorX = sqrt(segmentLocalErrorXX+trajectoryLocalErrorXX);

   return deltaX/errorX;
}

double MuonSegmentMatch::pullY() const {
   double deltaY, errorY;
   deltaY = segmentLocalPosition.Y()-trajectoryLocalPosition.Y();
   errorY = sqrt(segmentLocalErrorYY+trajectoryLocalErrorYY);

   return deltaY/errorY;
}
