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
  unsigned int MaxNumTracksThreshold_;
  double minPtTight_;

public:
  HITrackFilterForPVFinding(const edm::ParameterSet& conf) : TrackFilterForPVFinding(conf) {
    NumTracksThreshold_ = conf.getParameter<int>("numTracksThreshold");
    MaxNumTracksThreshold_ = conf.getParameter<int>("maxNumTracksThreshold");
    minPtTight_ = conf.getParameter<double>("minPtTight");
    //std::cout << "HITrackFilterForPVFinding  numTracksThreshold="<< NumTracksThreshold_ <<  std::endl;
  }

  // override the select method
  std::vector<reco::TransientTrack> select(const std::vector<reco::TransientTrack>& tracks) const override {
    std::vector<reco::TransientTrack> seltks = TrackFilterForPVFinding::select(tracks);
    if (seltks.size() < NumTracksThreshold_) {
      return tracks;
    } else if (seltks.size() > MaxNumTracksThreshold_) {
      std::vector<reco::TransientTrack> seltksTight = TrackFilterForPVFinding::selectTight(tracks, minPtTight_);
      if (seltksTight.size() >= NumTracksThreshold_)
        return seltksTight;
    }

    return seltks;
  }

  static void fillPSetDescription(edm::ParameterSetDescription& desc) {
    TrackFilterForPVFinding::fillPSetDescription(desc);
    desc.add<int>("numTracksThreshold", 0);  // HI only
    desc.add<int>("maxNumTracksThreshold", std::numeric_limits<int>::max());
    desc.add<double>("minPtTight", 0.0);
  }
};

#endif
