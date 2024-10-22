#ifndef TRACKINGTOOLS_TRANSIENTRACKBUILDER_H
#define TRACKINGTOOLS_TRANSIENTRACKBUILDER_H

#include "TrackingTools/TransientTrack/interface/TransientTrack.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrackFwd.h"
#include "DataFormats/Candidate/interface/CandidateFwd.h"
#include "DataFormats/Common/interface/ValueMap.h"

/**
   * Helper class to build TransientTrack from the persistent Track.
   * This is obtained from the eventSetup, as given in the example in the test
   * directory.
   */

class TransientTrackBuilder {
public:
  TransientTrackBuilder(const MagneticField* field, const edm::ESHandle<GlobalTrackingGeometry>& trackingGeometry)
      : theField(field), theTrackingGeometry(trackingGeometry) {}

  reco::TransientTrack build(const reco::Track* p) const;
  reco::TransientTrack build(const reco::Track& p) const;
  reco::TransientTrack build(const reco::GsfTrack* p) const;
  reco::TransientTrack build(const reco::GsfTrack& p) const;

  reco::TransientTrack build(const reco::TrackRef* p) const;
  reco::TransientTrack build(const reco::TrackRef& p) const;
  reco::TransientTrack build(const reco::GsfTrackRef* p) const;
  reco::TransientTrack build(const reco::GsfTrackRef& p) const;

  reco::TransientTrack build(const reco::CandidatePtr* p) const;
  reco::TransientTrack build(const reco::CandidatePtr& p) const;

  std::vector<reco::TransientTrack> build(const edm::Handle<reco::TrackCollection>& trkColl) const;
  std::vector<reco::TransientTrack> build(const edm::Handle<reco::GsfTrackCollection>& trkColl) const;
  std::vector<reco::TransientTrack> build(const edm::Handle<edm::View<reco::Track> >& trkColl) const;

  std::vector<reco::TransientTrack> build(const edm::Handle<reco::TrackCollection>& trkColl,
                                          const edm::ValueMap<float>& trackTimes,
                                          const edm::ValueMap<float>& trackTimeResos) const;
  std::vector<reco::TransientTrack> build(const edm::Handle<reco::GsfTrackCollection>& trkColl,
                                          const edm::ValueMap<float>& trackTimes,
                                          const edm::ValueMap<float>& trackTimeResos) const;
  std::vector<reco::TransientTrack> build(const edm::Handle<edm::View<reco::Track> >& trkColl,
                                          const edm::ValueMap<float>& trackTimes,
                                          const edm::ValueMap<float>& trackTimeResos) const;

  std::vector<reco::TransientTrack> build(const edm::Handle<reco::TrackCollection>& trkColl,
                                          const reco::BeamSpot& beamSpot) const;
  std::vector<reco::TransientTrack> build(const edm::Handle<reco::GsfTrackCollection>& trkColl,
                                          const reco::BeamSpot& beamSpot) const;
  std::vector<reco::TransientTrack> build(const edm::Handle<edm::View<reco::Track> >& trkColl,
                                          const reco::BeamSpot& beamSpot) const;

  std::vector<reco::TransientTrack> build(const edm::Handle<reco::TrackCollection>& trkColl,
                                          const reco::BeamSpot& beamSpot,
                                          const edm::ValueMap<float>& trackTimes,
                                          const edm::ValueMap<float>& trackTimeResos) const;
  std::vector<reco::TransientTrack> build(const edm::Handle<reco::GsfTrackCollection>& trkColl,
                                          const reco::BeamSpot& beamSpot,
                                          const edm::ValueMap<float>& trackTimes,
                                          const edm::ValueMap<float>& trackTimeResos) const;
  std::vector<reco::TransientTrack> build(const edm::Handle<edm::View<reco::Track> >& trkColl,
                                          const reco::BeamSpot& beamSpot,
                                          const edm::ValueMap<float>& trackTimes,
                                          const edm::ValueMap<float>& trackTimeResos) const;

  reco::TransientTrack build(const FreeTrajectoryState& fts) const;

  const MagneticField* field() const { return theField; }
  const edm::ESHandle<GlobalTrackingGeometry> trackingGeometry() const { return theTrackingGeometry; }
  static constexpr float defaultInvalidTrackTimeReso = 0.350f;

private:
  const MagneticField* theField;
  edm::ESHandle<GlobalTrackingGeometry> theTrackingGeometry;
};

#endif
