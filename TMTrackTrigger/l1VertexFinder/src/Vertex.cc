
#include "TMTrackTrigger/l1VertexFinder/interface/Vertex.h"



namespace vertexFinder {

void Vertex::computeParameters(){
	pT_ = 0.;
	z0_ = 0.;
	met_ = 0.;
	metX_ = 0.;
	metY_ = 0.;
	float z0square = 0.;
	for(TP track : tracks_){
		pT_ += track.pt();
		z0_ += track.z0();
		z0square += track.z0()*track.z0();
		metX_ += track.pt()*cos(track.phi0());
		metY_ += track.pt()*sin(track.phi0());
	}
	met_ = sqrt(metX_*metX_ + metY_*metY_);
	z0_ /= tracks_.size();
	z0square /= tracks_.size();
	z0width_ = sqrt(fabs(z0_*z0_ - z0square));
}

} // end namespace vertexFinder