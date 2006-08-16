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

double MuonSegmentMatch::pullX(bool scaled) const {
	double deltaX, errorX;
	deltaX = segmentLocalPosition.X()-trajectoryLocalPosition.X();

	if(scaled)
		errorX = sqrt(scaledSegmentLocalErrorXX()+scaledTrajectoryLocalErrorXX());
	else 
		errorX = sqrt(segmentLocalErrorXX+trajectoryLocalErrorXX);

	return deltaX/errorX;
}

double MuonSegmentMatch::pullY(bool scaled) const {
	double deltaY, errorY;
	deltaY = segmentLocalPosition.Y()-trajectoryLocalPosition.Y();

	if(scaled)
		errorY = sqrt(scaledSegmentLocalErrorYY()+scaledTrajectoryLocalErrorYY());
	else 
		errorY = sqrt(segmentLocalErrorYY+trajectoryLocalErrorYY);

	return deltaY/errorY;
}

double MuonSegmentMatch::scaledSegmentLocalErrorXX() const {
	double tanTheta = segmentLocalDirection.X()/segmentLocalDirection.Z();
	double cosTheta2 = 1/(1+tanTheta*tanTheta);
	return segmentLocalErrorXX/(cosTheta2*cosTheta2);
}

double MuonSegmentMatch::scaledTrajectoryLocalErrorXX() const {
	double tanTheta = trajectoryLocalDirection.X()/trajectoryLocalDirection.Z();
	double cosTheta2 = 1/(1+tanTheta*tanTheta);
	return trajectoryLocalErrorXX/(cosTheta2*cosTheta2);
}

double MuonSegmentMatch::scaledSegmentLocalErrorYY() const {
	double tanTheta = segmentLocalDirection.Y()/segmentLocalDirection.Z();
	double cosTheta2 = 1/(1+tanTheta*tanTheta);
	return segmentLocalErrorYY/(cosTheta2*cosTheta2);
}

double MuonSegmentMatch::scaledTrajectoryLocalErrorYY() const {
	double tanTheta = trajectoryLocalDirection.Y()/trajectoryLocalDirection.Z();
	double cosTheta2 = 1/(1+tanTheta*tanTheta);
	return trajectoryLocalErrorYY/(cosTheta2*cosTheta2);
}
