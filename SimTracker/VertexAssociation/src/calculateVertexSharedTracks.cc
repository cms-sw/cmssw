#include "SimTracker/VertexAssociation/interface/calculateVertexSharedTracks.h"

unsigned int calculateVertexSharedTracks(const reco::Vertex& recoV, const TrackingVertex& simV, const reco::RecoToSimCollection& trackRecoToSimAssociation) {
  unsigned int sharedTracks = 0;
  for(auto iTrack = recoV.tracks_begin(); iTrack != recoV.tracks_end(); ++iTrack) {
    auto found = trackRecoToSimAssociation.find(*iTrack);

    if(found == trackRecoToSimAssociation.end())
      continue;

    // matched TP equal to any TP of sim vertex => increase counter
    for(const auto& tp: found->val) {
      if(std::find_if(simV.daughterTracks_begin(), simV.daughterTracks_end(), [&](const TrackingParticleRef& vtp) {
            return tp.first == vtp;
          }) != simV.daughterTracks_end()) {
        sharedTracks += 1;
        break;
      }
    }
  }

  return sharedTracks;
}

unsigned int calculateVertexSharedTracks(const TrackingVertex& simV, const reco::Vertex& recoV, const reco::SimToRecoCollection& trackSimToRecoAssociation) {
  unsigned int sharedTracks = 0;
  for(auto iTP = simV.daughterTracks_begin(); iTP != simV.daughterTracks_end(); ++iTP) {
    auto found = trackSimToRecoAssociation.find(*iTP);

    if(found == trackSimToRecoAssociation.end())
      continue;

    // matched track equal to any track of reco vertex => increase counter
    for(const auto& tk: found->val) {
      if(std::find_if(recoV.tracks_begin(), recoV.tracks_end(), [&](const reco::TrackBaseRef& vtk) {
            return tk.first == vtk;
          }) != recoV.tracks_end()) {
        sharedTracks += 1;
        break;
      }
    }
  }

  return sharedTracks;
}
