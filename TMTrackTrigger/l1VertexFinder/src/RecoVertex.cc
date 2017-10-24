#include "TMTrackTrigger/VertexFinder/interface/RecoVertex.h"


namespace vertexFinder {

void RecoVertex::computeParameters(){
	pT_ = 0.;
	z0_ = 0.;
	met_ = 0.;
	metX_ = 0.;
	metY_ = 0.;
	float z0square = 0.;
	highPt_ = false;
	numHighPtTracks_ = 0;
	for(const L1fittedTrack* track : tracks_){
		pT_ += track->pt();
		z0_ += track->z0();
		z0square += track->z0()*track->z0();
		
		if(track->pt() > 200.){
			metX_ += 0.*cos(track->phi0());
			metY_ += 0.*sin(track->phi0());
		} else{
			metX_ += track->pt()*cos(track->phi0());
			metY_ += track->pt()*sin(track->phi0());
		}
		if(track->pt()>10.){
			highPt_ = true;
			numHighPtTracks_++;
		}
	}
	z0_ /= tracks_.size();
	met_ = sqrt(metX_*metX_ + metY_*metY_);
	z0square /= tracks_.size();
	z0width_ = sqrt(fabs(z0_*z0_ - z0square));
}

}
