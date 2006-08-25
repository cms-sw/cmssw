#include "TrackingTools/TrackAssociator/interface/MuonSegmentMatch.h"
#include "DataFormats/MuonDetId/interface/DTChamberId.h"
#include "DataFormats/MuonDetId/interface/CSCDetId.h"
#include "DataFormats/MuonDetId/interface/RPCDetId.h"

int MuonSegmentMatch::station() const {
	int muonSubdetId = id.subdetId();

	if(muonSubdetId==1) {//DT
		DTChamberId segId(id.rawId());
		return segId.station();
	}
	if(muonSubdetId==2) {//CSC
		CSCDetId segId(id.rawId());
		return segId.station();
	}
	if(muonSubdetId==3) {//RPC
		RPCDetId segId(id.rawId());
		return segId.station();
	}

	return -1;
}

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
