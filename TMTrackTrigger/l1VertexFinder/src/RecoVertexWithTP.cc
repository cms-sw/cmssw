#include "TMTrackTrigger/l1VertexFinder/interface/RecoVertexWithTP.h"

namespace l1tVertexFinder {

  RecoVertexWithTP::RecoVertexWithTP(
    RecoVertex & vertex,
    std::vector<L1fittedTrack> l1Tracks)
  {
    z0_ = -999.;
    pT_ = -9999.;
    met_ = -999.;

    // create a map for associating fat reco tracks with their underlying
    // TTTrack pointers
    // std::map <const edm::Ptr<TTTrack< Ref_Phase2TrackerDigi_ >>, L1fittedTrack> trackAssociationMap;
    std::map <const edm::Ptr<TTTrack< Ref_Phase2TrackerDigi_ >>, const L1fittedTrack *> trackAssociationMap;

    // unsigned int index = 0;
    // get a list of reconstructed tracks with references to their TPs
    for (const auto & trackIt: l1Tracks) {
      trackAssociationMap.insert(std::pair<const edm::Ptr<TTTrack< Ref_Phase2TrackerDigi_ >>, const L1fittedTrack *>(trackIt.getTTTrackPtr(), &trackIt));
    }

    // loop over base fitted tracks in reco vertex and find the corresponding TP
    // track using the TTTrack - L1fittedTrack map from above
    for (const auto & trackIt : vertex.tracks()) {
      tracks_.emplace_back(trackAssociationMap[trackIt->getTTTrackPtr()]);
    }
  }

  void RecoVertexWithTP::computeParameters(bool weightedmean){
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
    for(const L1fittedTrackBase* track : tracks_){
      // if(track->pt() < 100.){
      pT_ += track->pt();
      SumZ += track->z0();
      SumZ_pT += track->z0()*track->pt();
      z0square += track->z0()*track->z0();
      // if(track->getNumStubs() > 4){
      /*
	if(track->pt() > 999. and track->getNumStubs() < 6){
	metX_ += 100.*cos(track->phi0());
	metY_ += 100.*sin(track->phi0());
	} else{
	metX_ += track->pt()*cos(track->phi0());
	metY_ += track->pt()*sin(track->phi0());
	}
      */
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
