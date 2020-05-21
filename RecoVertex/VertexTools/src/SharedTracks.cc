#include "RecoVertex/VertexTools/interface/SharedTracks.h"
namespace vertexTools {
  using namespace reco;
  double computeSharedTracks(const Vertex &pv, const std::vector<TrackRef> &svTracks, double minTrackWeight, float) {
    std::set<TrackRef> pvTracks;
    for (std::vector<TrackBaseRef>::const_iterator iter = pv.tracks_begin(); iter != pv.tracks_end(); iter++)
      if (pv.trackWeight(*iter) >= minTrackWeight)
        pvTracks.insert(iter->castTo<TrackRef>());

    unsigned int count = 0;
    for (const auto &svTrack : svTracks)
      count += pvTracks.count(svTrack);

    return (double)count / (double)svTracks.size();
  }
  double computeSharedTracks(const Vertex &pv,
                             const std::vector<CandidatePtr> &svTracks,
                             double minTrackWeight,
                             float maxsigma) {
    unsigned int count = 0;
    for (const auto &svTrack : svTracks) {
      if (std::abs(svTrack->bestTrack()->dz() - pv.z()) / svTrack->bestTrack()->dzError() < maxsigma &&
          std::abs(svTrack->bestTrack()->dxy(pv.position()) / svTrack->bestTrack()->dxyError()) < maxsigma)
        count++;
    }
    return (double)count / (double)svTracks.size();
  }
  double computeSharedTracks(const VertexCompositePtrCandidate &sv2,
                             const std::vector<CandidatePtr> &svTracks,
                             double,
                             float) {
    unsigned int count = 0;
    for (const auto &svTrack : svTracks) {
      if (std::find(sv2.daughterPtrVector().begin(), sv2.daughterPtrVector().end(), svTrack) !=
          sv2.daughterPtrVector().end())
        count++;
    }
    return (double)count / (double)svTracks.size();
  }

  double computeSharedTracks(const reco::Vertex &pv,
                             const reco::VertexCompositePtrCandidate &sv,
                             double minTrackWeight,
                             float mindist) {
    return computeSharedTracks(pv, sv.daughterPtrVector(), minTrackWeight, mindist);
  }
  double computeSharedTracks(const reco::Vertex &pv, const reco::Vertex &sv, double minTrackWeight, float) {
    std::vector<TrackRef> svTracks;
    for (std::vector<TrackBaseRef>::const_iterator iter = sv.tracks_begin(); iter != sv.tracks_end(); iter++)
      if (sv.trackWeight(*iter) >= minTrackWeight)
        svTracks.push_back(iter->castTo<TrackRef>());
    return computeSharedTracks(pv, svTracks, minTrackWeight);
  }
  double computeSharedTracks(const reco::VertexCompositePtrCandidate &sv,
                             const reco::VertexCompositePtrCandidate &sv2,
                             double,
                             float) {
    return computeSharedTracks(sv, sv2.daughterPtrVector());
  }

}  // namespace vertexTools
