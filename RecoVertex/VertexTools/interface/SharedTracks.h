#ifndef SharedTracks_h
#define SharedTracks_h
#include <vector>
#include "DataFormats/Candidate/interface/VertexCompositePtrCandidate.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/Candidate/interface/Candidate.h"

namespace vertexTools {
	double computeSharedTracks(const reco::Vertex &pv, const std::vector<reco::TrackRef> &svTracks, double minTrackWeight=0.5, float unused=0);
	double computeSharedTracks(const reco::Vertex &pv, const std::vector<reco::CandidatePtr> &svTracks, double minTrackWeight=0.5,float mindist=2.0);
	double computeSharedTracks(const reco::Vertex &pv, const reco::VertexCompositePtrCandidate &sv, double minTrackWeight=0.5,float mindist=2.0);
	double computeSharedTracks(const reco::Vertex &pv, const reco::Vertex &sv, double minTrackWeight=0.5,float mindist=2.0);
	double computeSharedTracks(const reco::VertexCompositePtrCandidate &sv, const reco::VertexCompositePtrCandidate &sv2, double minTrackWeight=0.5,float mindist=2.0);
        double computeSharedTracks(const reco::VertexCompositePtrCandidate &sv2, const std::vector<reco::CandidatePtr> &svTracks, double unused1=0, float unused2=0 );

}
#endif
