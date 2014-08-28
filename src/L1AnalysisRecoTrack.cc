#include "L1TriggerDPG/L1Ntuples/interface/L1AnalysisRecoTrack.h"



void L1Analysis::L1AnalysisRecoTrack::SetTracks(const reco::TrackCollection& trackColl, unsigned maxTrack)
{
   
   track_.nTrk = trackColl.size();
   
   reco::TrackBase::TrackQuality hiPurity = reco::TrackBase::qualityByName("highPurity");
    for(reco::TrackCollection::const_iterator itk = trackColl.begin();
	itk!=trackColl.end();
	++itk){
      if(itk->quality(hiPurity)) track_.nHighPurity++;
    }
    track_.fHighPurity = static_cast<float>(track_.nHighPurity)/static_cast<float>(track_.nTrk);

}

