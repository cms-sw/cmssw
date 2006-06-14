#include "TrackingTools/TransientTrack/interface/TransientTrackBuilder.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"

using namespace reco;
using namespace std;

TransientTrack* TransientTrackBuilder::build (const Track * t) const {

  return new TransientTrack(*t, theField);

}

TransientTrack* TransientTrackBuilder::build (const TrackRef * t) const {

  return new TransientTrack(*t, theField);

}

vector<TransientTrack> 
TransientTrackBuilder::build ( const edm::Handle<reco::TrackCollection> & trkColl)  const {

  vector<TransientTrack> ttVect;
  ttVect.reserve((*trkColl).size());
  for (unsigned int i = 0; i < (*trkColl).size() ; i++) {
    ttVect.push_back(TransientTrack(TrackRef(trkColl, i), theField));
  }
  return ttVect;

}

