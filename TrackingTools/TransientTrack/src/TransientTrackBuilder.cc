#include "TrackingTools/TransientTrack/interface/TransientTrackBuilder.h"
#include "DataFormats/Common/interface/Handle.h" 
#include "TrackingTools/TransientTrack/interface/GsfTransientTrack.h"
#include "TrackingTools/TransientTrack/interface/TransientTrackFromFTS.h"
#include "FWCore/Utilities/interface/isFinite.h"


using namespace reco;
using namespace std;
using namespace edm;

namespace {
  constexpr float fakeBeamSpotTimeWidth = 0.175f;
}

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

TransientTrack TransientTrackBuilder::build (const CandidatePtr * t) const {
  return TransientTrack(*t, theField, theTrackingGeometry);
}

TransientTrack TransientTrackBuilder::build (const CandidatePtr & t) const {
  return TransientTrack(t, theField, theTrackingGeometry);
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
TransientTrackBuilder::build (const edm::Handle<edm::View<Track> > & trkColl) const
{
  vector<TransientTrack> ttVect;
  ttVect.reserve((*trkColl).size());
  for (unsigned int i = 0; i < (*trkColl).size() ; i++) {
    const Track * trk = &(*trkColl)[i];
    const GsfTrack * gsfTrack = dynamic_cast<const GsfTrack *>(trk);
    if (gsfTrack) {
      ttVect.push_back( TransientTrack(
	  new GsfTransientTrack(RefToBase<Track>(trkColl, i).castTo<GsfTrackRef>(), theField, theTrackingGeometry)) );
    } else { // gsf
      ttVect.push_back(TransientTrack(RefToBase<Track>(trkColl, i).castTo<TrackRef>(), theField, theTrackingGeometry));
    }
  }
  return ttVect;
}

vector<TransientTrack> 
TransientTrackBuilder::build ( const edm::Handle<reco::TrackCollection> & trkColl,
			       const edm::ValueMap<float>& trackTimes,
                               const edm::ValueMap<float>& trackTimeResos ) const
{
  vector<TransientTrack> ttVect;
  ttVect.reserve((*trkColl).size());  
  for (unsigned int i = 0; i < (*trkColl).size() ; i++) {
    TrackRef ref(trkColl, i);
    double time = trackTimes[ref];
    double timeReso = trackTimeResos[ref];
    timeReso = ( timeReso > 1e-6 ? timeReso : 2.0*fakeBeamSpotTimeWidth ); // make the error much larger than the BS time width
    if( edm::isNotFinite(time) ) {
      time = 0.0;
      timeReso = 2.0*fakeBeamSpotTimeWidth;
    }
    ttVect.push_back(TransientTrack(ref, time, timeReso, theField, theTrackingGeometry));
  }
  return ttVect;
}

vector<TransientTrack> 
TransientTrackBuilder::build (const edm::Handle<reco::GsfTrackCollection> & trkColl,
			      const edm::ValueMap<float>& trackTimes,
			      const edm::ValueMap<float>& trackTimeResos ) const
{
  vector<TransientTrack> ttVect;
  ttVect.reserve((*trkColl).size());  
  for (unsigned int i = 0; i < (*trkColl).size() ; i++) {
    GsfTrackRef ref(trkColl, i);
    double time = trackTimes[ref];
    double timeReso = trackTimeResos[ref];
    timeReso = ( timeReso > 1e-6 ? timeReso : 2.0*fakeBeamSpotTimeWidth ); // make the error much larger than the BS time width
    if( edm::isNotFinite(time) ) {
      time = 0.0;
      timeReso = 2.0*fakeBeamSpotTimeWidth;
    }
    ttVect.push_back( TransientTrack(
	new GsfTransientTrack(ref, time, timeReso, theField, theTrackingGeometry)) );
  }
  return ttVect;
}

vector<TransientTrack> 
TransientTrackBuilder::build (const edm::Handle<edm::View<Track> > & trkColl,
			      const edm::ValueMap<float>& trackTimes,
			      const edm::ValueMap<float>& trackTimeResos ) const
{
  vector<TransientTrack> ttVect;
  ttVect.reserve((*trkColl).size());  
  for (unsigned int i = 0; i < (*trkColl).size() ; i++) {
    const Track * trk = &(*trkColl)[i];
    const GsfTrack * gsfTrack = dynamic_cast<const GsfTrack *>(trk);
    if (gsfTrack) {
      GsfTrackRef ref = RefToBase<Track>(trkColl, i).castTo<GsfTrackRef>();
      double time = trackTimes[ref];
      double timeReso = trackTimeResos[ref];
      timeReso = ( timeReso > 1e-6 ? timeReso : 2.0*fakeBeamSpotTimeWidth ); // make the error much larger than the BS time width
      if( edm::isNotFinite(time) ) {
	time = 0.0;
	timeReso = 2.0*fakeBeamSpotTimeWidth;
      }
      ttVect.push_back( TransientTrack(
	  new GsfTransientTrack(RefToBase<Track>(trkColl, i).castTo<GsfTrackRef>(), time, timeReso, theField, theTrackingGeometry)) );
    } else { // gsf
      TrackRef ref = RefToBase<Track>(trkColl, i).castTo<TrackRef>();
      double time = trackTimes[ref];
      double timeReso = trackTimeResos[ref];
      timeReso = ( timeReso > 1e-6 ? timeReso : 2.0*fakeBeamSpotTimeWidth ); // make the error much larger than the BS time width
      if( edm::isNotFinite(time) ) {
	time = 0.0;
	timeReso = 2.0*fakeBeamSpotTimeWidth;
      }
      ttVect.push_back(TransientTrack(RefToBase<Track>(trkColl, i).castTo<TrackRef>(), time, timeReso, theField, theTrackingGeometry));
    }
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

vector<TransientTrack> 
TransientTrackBuilder::build (const edm::Handle<edm::View<Track> > & trkColl,
	const reco::BeamSpot & beamSpot) const
{
  vector<TransientTrack> ttVect = build(trkColl);
  for (unsigned int i = 0; i < ttVect.size() ; i++) {
    ttVect[i].setBeamSpot(beamSpot);
  }
  return ttVect;
}

vector<TransientTrack> 
TransientTrackBuilder::build ( const edm::Handle<reco::TrackCollection> & trkColl,
			       const reco::BeamSpot & beamSpot,
			       const edm::ValueMap<float>& trackTimes,
			       const edm::ValueMap<float>& trackTimeResos ) const
{
  vector<TransientTrack> ttVect = build(trkColl, trackTimes, trackTimeResos );
  for (unsigned int i = 0; i < ttVect.size() ; i++) {
    ttVect[i].setBeamSpot(beamSpot);
  }
  return ttVect;
}

vector<TransientTrack> 
TransientTrackBuilder::build ( const edm::Handle<reco::GsfTrackCollection> & trkColl,
			       const reco::BeamSpot & beamSpot,
			       const edm::ValueMap<float>& trackTimes,
			       const edm::ValueMap<float>& trackTimeResos ) const
{
  vector<TransientTrack> ttVect = build(trkColl, trackTimes, trackTimeResos);
  for (unsigned int i = 0; i < ttVect.size() ; i++) {
    ttVect[i].setBeamSpot(beamSpot);
  }
  return ttVect;
}

vector<TransientTrack> 
TransientTrackBuilder::build ( const edm::Handle<edm::View<Track> > & trkColl,
			       const reco::BeamSpot & beamSpot,
			       const edm::ValueMap<float>& trackTimes,
			       const edm::ValueMap<float>& trackTimeResos ) const
{
  vector<TransientTrack> ttVect = build(trkColl, trackTimes, trackTimeResos);
  for (unsigned int i = 0; i < ttVect.size() ; i++) {
    ttVect[i].setBeamSpot(beamSpot);
  }
  return ttVect;
}

TransientTrack TransientTrackBuilder::build (const FreeTrajectoryState & fts) const {
  return TransientTrack(new TransientTrackFromFTS(fts));
}
