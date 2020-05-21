
#include "SimTracker/TrackHistory/interface/JetVetoedTracksAssociatorDRVertex.h"

JetVetoedTracksAssociationDRVertex::JetVetoedTracksAssociationDRVertex(double dr) : mDeltaR2Threshold(dr * dr) {}

void JetVetoedTracksAssociationDRVertex::produce(reco::JetTracksAssociation::Container *fAssociation,
                                                 const std::vector<edm::RefToBase<reco::Jet>> &fJets,
                                                 const std::vector<reco::TrackRef> &fTracks,
                                                 TrackClassifier &classifier) const {
  // cache tracks kinematics
  std::vector<math::RhoEtaPhiVector> trackP3s;
  trackP3s.reserve(fTracks.size());
  for (const auto &fTrack : fTracks) {
    const reco::Track *track = &*fTrack;
    trackP3s.push_back(math::RhoEtaPhiVector(track->p(), track->eta(), track->phi()));
  }
  // loop on jets and associate
  for (const auto &fJet : fJets) {
    reco::TrackRefVector assoTracks;
    const reco::Jet *jet = &*fJet;
    double jetEta = jet->eta();
    double jetPhi = jet->phi();
    for (unsigned t = 0; t < fTracks.size(); ++t) {
      double dR2 = deltaR2(jetEta, jetPhi, trackP3s[t].eta(), trackP3s[t].phi());
      classifier.evaluate(reco::TrackBaseRef(fTracks[t]));
      if (dR2 < mDeltaR2Threshold &&
          (classifier.is(TrackClassifier::BWeakDecay) || classifier.is(TrackClassifier::CWeakDecay) ||
           classifier.is(TrackClassifier::PrimaryVertex)))
        assoTracks.push_back(fTracks[t]);
    }
    reco::JetTracksAssociation::setValue(fAssociation, fJet, assoTracks);
  }
}
