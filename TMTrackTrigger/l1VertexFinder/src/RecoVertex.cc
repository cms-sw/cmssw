
#include "TMTrackTrigger/l1VertexFinder/interface/RecoVertex.h"



namespace l1tVertexFinder {

void RecoVertex::computeParameters(bool weightedmean){
	pT_ = 0.;
	z0_ = 0.;
	met_ = 0.;
	metX_ = 0.;
	metY_ = 0.;
	float z0square = 0.;
	highPt_ = false;
	highestPt_ = 0.;
	numHighPtTracks_ = 0;
	unsigned int overflows = 0;
	float SumZ_pT = 0.;
	float SumZ = 0.;
	for(const L1fittedTrack* track : tracks_){
		// if(track->pt() < 100.){
			pT_ += track->pt();
			SumZ += track->z0();
			SumZ_pT += track->z0()*track->pt();
			z0square += track->z0()*track->z0();
			// if(track->getNumStubs() > 4){
			if(track->pt() > 999. and track->getNumStubs() < 6){
				metX_ += 100.*cos(track->phi0());
				metY_ += 100.*sin(track->phi0());
			} else{
				metX_ += track->pt()*cos(track->phi0());
				metY_ += track->pt()*sin(track->phi0());
			}
			// }
			if(track->pt()>15.){
			 	highPt_ = true;
			 	highestPt_ = track->pt();
			// 	numHighPtTracks_++;
			}
		// } else{
		// 	overflows++;
		// }
	}
	// unsigned int divider = tracks_.size() - overflows;
	// if(divider > 0)	{

	if(weightedmean){
		z0_ = SumZ_pT/pT_;
	} else{
		z0_ = SumZ/tracks_.size();
	}	

	met_ = sqrt(metX_*metX_ + metY_*metY_);
	z0square /= tracks_.size();
	z0width_ = sqrt(fabs(z0_*z0_ - z0square));

}

} // end ns l1tVertexFinder
