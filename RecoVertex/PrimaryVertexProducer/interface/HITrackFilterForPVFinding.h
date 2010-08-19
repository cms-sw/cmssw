#ifndef HITrackFilterForPVFinding_h
#define HITrackFilterForPVFinding_h

/**\class HITrackFilterForPVFinding 
 
  Description: selects tracks for primary vertex reconstruction using th TrackFilterForPVFinding,
  returns the input set of tracks if less than NumTracksThreshold tracks were selected

*/

#include "RecoVertex/PrimaryVertexProducer/interface/TrackFilterForPVFinding.h"


class HITrackFilterForPVFinding : public TrackFilterForPVFinding {

private:
  unsigned int NumTracksThreshold_;


 public:

  HITrackFilterForPVFinding(const edm::ParameterSet& conf):TrackFilterForPVFinding(conf){
    NumTracksThreshold_=conf.getParameter<int>("numTracksThreshold");
    //std::cout << "HITrackFilterForPVFinding  numTracksThreshold="<< NumTracksThreshold_ <<  std::endl;
  }
    

  // override the select method
  std::vector<reco::TransientTrack> select (const std::vector<reco::TransientTrack>& tracks) const{
    std::vector<reco::TransientTrack> seltks = TrackFilterForPVFinding::select(tracks);
    if (seltks.size()<NumTracksThreshold_){
      return tracks;
    }else{
      return seltks;
    }
  }


};

#endif
