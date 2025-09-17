#include "SimTracker/VertexAssociation/interface/calculateVertexSharedTracks.h"

SharedTracksAndFractions calculateVertexSharedTracks(const reco::Vertex &recoV,
                                                     const TrackingVertex &simV,
                                                     const reco::RecoToSimCollection &trackRecoToSimAssociation) {
  unsigned int nSharedTracks = 0;
  float sharedTracksWeightPtSum2 = 0;
  float totalTracksWeightPtSum2 = 0;
  float sharedTracksWeightDzError = 0;
  float totalTracksWeightDzError = 0;

  for (auto iTrack = recoV.tracks_begin(); iTrack != recoV.tracks_end(); ++iTrack) {
    auto found = trackRecoToSimAssociation.find(*iTrack);

    totalTracksWeightDzError += 1.0 / ((*iTrack)->dzError() * (*iTrack)->dzError());
    totalTracksWeightPtSum2 += (*iTrack)->pt() * (*iTrack)->pt();

    if (found == trackRecoToSimAssociation.end())
      continue;

    // matched TP equal to any TP of sim vertex => increase counter
    for (const auto &tp : found->val) {
      if (std::find_if(simV.daughterTracks_begin(), simV.daughterTracks_end(), [&](const TrackingParticleRef &vtp) {
            return tp.first == vtp;
          }) != simV.daughterTracks_end()) {
        nSharedTracks += 1;
        sharedTracksWeightDzError += 1.0 / ((*iTrack)->dzError() * (*iTrack)->dzError());
        sharedTracksWeightPtSum2 += ((*iTrack)->pt() * (*iTrack)->pt());
        break;
      }
    }
  }

  float sharedTracksFraction = (recoV.tracksSize() > 0) ? (float(nSharedTracks) / recoV.tracksSize()) : 0.0f;
  float sharedPt2Fraction = (totalTracksWeightPtSum2 > 0) ? (sharedTracksWeightPtSum2 / totalTracksWeightPtSum2) : 0.0f;
  float sharedDzErrFraction =
      (totalTracksWeightDzError > 0) ? (sharedTracksWeightDzError / totalTracksWeightDzError) : 0.0f;

  return SharedTracksAndFractions(nSharedTracks, sharedTracksFraction, sharedPt2Fraction, sharedDzErrFraction);
}

SharedTracksAndFractions calculateVertexSharedTracks(const TrackingVertex &simV,
                                                     const reco::Vertex &recoV,
                                                     const reco::SimToRecoCollection &trackSimToRecoAssociation) {
  unsigned int nSharedTracks = 0;
  float sharedTracksWeightPtSum2 = 0;
  float totalTracksWeightPtSum2 = 0;
  float sharedTracksWeightDzError = 0;
  float totalTracksWeightDzError = 0;

  for (auto iTrack = recoV.tracks_begin(); iTrack != recoV.tracks_end(); ++iTrack) {
    totalTracksWeightPtSum2 += ((*iTrack)->pt() * (*iTrack)->pt());
    totalTracksWeightDzError += 1.0 / ((*iTrack)->dzError() * (*iTrack)->dzError());
  }

  for (auto iTP = simV.daughterTracks_begin(); iTP != simV.daughterTracks_end(); ++iTP) {
    auto found = trackSimToRecoAssociation.find(*iTP);

    if (found == trackSimToRecoAssociation.end())
      continue;

    // matched track equal to any track of reco vertex => increase counter
    for (const auto &tk : found->val) {
      if (std::find_if(recoV.tracks_begin(), recoV.tracks_end(), [&](const reco::TrackBaseRef &vtk) {
            totalTracksWeightPtSum2 += (tk.first->pt() * tk.first->pt());
            return ((tk.first.id() == vtk.id()) &&
                    (tk.first.key() == vtk.key()));  // tk.first == vtk; operator::== not working
          }) != recoV.tracks_end()) {
        nSharedTracks += 1;
        sharedTracksWeightDzError += 1.0 / (tk.first->dzError() * tk.first->dzError());
        sharedTracksWeightPtSum2 += (tk.first->pt() * tk.first->pt());
        break;
      }
    }
  }

  float sharedTracksFraction = (recoV.tracksSize() > 0) ? (float(nSharedTracks) / recoV.tracksSize()) : 0.0f;
  float sharedPt2Fraction = (totalTracksWeightPtSum2 > 0) ? (sharedTracksWeightPtSum2 / totalTracksWeightPtSum2) : 0.0f;
  float sharedDzErrFraction =
      (totalTracksWeightDzError > 0) ? (sharedTracksWeightDzError / totalTracksWeightDzError) : 0.0f;

  return SharedTracksAndFractions(nSharedTracks, sharedTracksFraction, sharedPt2Fraction, sharedDzErrFraction);
}
