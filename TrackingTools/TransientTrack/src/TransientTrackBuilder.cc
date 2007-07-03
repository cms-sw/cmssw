#include "TrackingTools/TransientTrack/interface/TransientTrackBuilder.h"
#include "TrackingTools/TransientTrack/interface/GsfTransientTrack.h"
#include "TrackingTools/TransientTrack/interface/TransientTrackFromFTS.h"

using namespace reco;
using namespace std;

TransientTrack TransientTrackBuilder::build (const Track * t) const {
  return TransientTrack(*t, theField, theTrackingGeometry);
}

TransientTrack TransientTrackBuilder::build (const Track & t) const {
  return TransientTrack(t, theField, theTrackingGeometry);
}

TransientTrack TransientTrackBuilder::build (const GsfTrack * t) const {
  return TransientTrack(new GsfTransientTrack(*t, theField, theTrackingGeometry));
}

TransientTrack TransientTrackBuilder::build (const GsfTrack & t) const {
  return TransientTrack(new GsfTransientTrack(t, theField, theTrackingGeometry));
}

TransientTrack TransientTrackBuilder::build (const TrackRef * t) const {
  return TransientTrack(*t, theField, theTrackingGeometry);
}

TransientTrack TransientTrackBuilder::build (const TrackRef & t) const {
  return TransientTrack(t, theField, theTrackingGeometry);
}


TransientTrack TransientTrackBuilder::build (const GsfTrackRef * t) const {
  return TransientTrack(new GsfTransientTrack(*t, theField, theTrackingGeometry));
}

TransientTrack TransientTrackBuilder::build (const GsfTrackRef & t) const {
  return TransientTrack(new GsfTransientTrack(t, theField, theTrackingGeometry));
}

vector<TransientTrack> 
TransientTrackBuilder::build ( const edm::Handle<reco::TrackCollection> & trkColl) const
{
  vector<TransientTrack> ttVect;
  ttVect.reserve((*trkColl).size());
  for (unsigned int i = 0; i < (*trkColl).size() ; i++) {
    ttVect.push_back(TransientTrack(TrackRef(trkColl, i), theField, theTrackingGeometry));
  }
  return ttVect;
}

vector<TransientTrack> 
TransientTrackBuilder::build (const edm::Handle<reco::GsfTrackCollection> & trkColl) const
{
  vector<TransientTrack> ttVect;
  ttVect.reserve((*trkColl).size());
  for (unsigned int i = 0; i < (*trkColl).size() ; i++) {
    ttVect.push_back( TransientTrack(
	new GsfTransientTrack(GsfTrackRef(trkColl, i), theField, theTrackingGeometry)) );
  }
  return ttVect;
}

vector<TransientTrack> 
TransientTrackBuilder::build ( const edm::Handle<reco::TrackCollection> & trkColl,
	const reco::BeamSpot & beamSpot) const
{
  vector<TransientTrack> ttVect = build(trkColl);
  for (unsigned int i = 0; i < ttVect.size() ; i++) {
    ttVect[i].setBeamSpot(beamSpot);
  }
  return ttVect;
}

vector<TransientTrack> 
TransientTrackBuilder::build (const edm::Handle<reco::GsfTrackCollection> & trkColl,
	const reco::BeamSpot & beamSpot) const
{
  vector<TransientTrack> ttVect = build(trkColl);
  for (unsigned int i = 0; i < ttVect.size() ; i++) {
    ttVect[i].setBeamSpot(beamSpot);
  }
  return ttVect;
}

TransientTrack TransientTrackBuilder::build (const FreeTrajectoryState & fts) const {
  return TransientTrack(new TransientTrackFromFTS(fts));
}
