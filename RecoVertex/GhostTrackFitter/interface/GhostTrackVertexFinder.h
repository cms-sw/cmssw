#ifndef RecoBTag_GhostTrackVertexFinder_h
#define RecoBTag_GhostTrackVertexFinder_h

#include <memory>
#include <vector>
#include <set>

#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "DataFormats/GeometryVector/interface/GlobalVector.h"
#include "DataFormats/GeometryCommonDetAlgo/interface/GlobalError.h"

#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"

#include "TrackingTools/TransientTrack/interface/TransientTrack.h"

#include "RecoVertex/VertexPrimitives/interface/TransientVertex.h"
#include "RecoVertex/VertexPrimitives/interface/VertexFitter.h"
#include "RecoVertex/VertexPrimitives/interface/VertexReconstructor.h"

#include "RecoVertex/GhostTrackFitter/interface/GhostTrackFitter.h"
#include "RecoVertex/GhostTrackFitter/interface/GhostTrack.h"

namespace reco {

class GhostTrack;
class GhostTrackFitter;

class GhostTrackVertexFinder { // : public VertexReconstructor
    public:
	enum FitType {
		kAlwaysWithGhostTrack,
		kSingleTracksWithGhostTrack,
		kRefitGhostTrackWithVertices
	};

	GhostTrackVertexFinder();
	GhostTrackVertexFinder(double maxFitChi2, double mergeThreshold,
	                       double primcut, double seccut,
	                       FitType fitType);
	~GhostTrackVertexFinder();

	std::vector<TransientVertex>
		vertices(const reco::Vertex &primaryVertex,
		         const GlobalVector &direction,
		         double coneRadius,
		         const std::vector<TransientTrack> &tracks) const;

	std::vector<TransientVertex>
		vertices(const GlobalPoint &primaryPosition,
		         const GlobalError &primaryError,
		         const GlobalVector &direction,
		         double coneRadius,
		         const std::vector<TransientTrack> &tracks) const;

	std::vector<TransientVertex>
		vertices(const reco::Vertex &primaryVertex,
		         const GlobalVector &direction,
		         double coneRadius,
		         const reco::BeamSpot &beamSpot,
		         const std::vector<TransientTrack> &tracks) const;

	std::vector<TransientVertex>
		vertices(const GlobalPoint &primaryPosition,
		         const GlobalError &primaryError,
		         const GlobalVector &direction,
		         double coneRadius,
		         const reco::BeamSpot &beamSpot,
		         const std::vector<TransientTrack> &tracks) const;

	std::vector<TransientVertex>
		vertices(const reco::Vertex &primaryVertex,
		         const GlobalVector &direction,
		         double coneRadius,
		         const reco::BeamSpot &beamSpot,
		         const std::vector<TransientTrack> &primaries,
		         const std::vector<TransientTrack> &tracks) const;

	std::vector<TransientVertex>
		vertices(const GlobalPoint &primaryPosition,
		         const GlobalError &primaryError,
		         const GlobalVector &direction,
		         double coneRadius,
		         const reco::BeamSpot &beamSpot,
		         const std::vector<TransientTrack> &primaries,
		         const std::vector<TransientTrack> &tracks) const;

	std::vector<TransientVertex>
		vertices(const reco::Vertex &primaryVertex,
		         const reco::Track &ghostTrack,
		         const std::vector<TransientTrack> &tracks,
	                 const std::vector<float> &weights = std::vector<float>()) const;

	std::vector<TransientVertex>
		vertices(const reco::Vertex &primaryVertex,
		         const reco::Track &ghostTrack,
		         const reco::BeamSpot &beamSpot,
		         const std::vector<TransientTrack> &tracks,
	                 const std::vector<float> &weights = std::vector<float>()) const;

	std::vector<TransientVertex>
		vertices(const reco::Vertex &primaryVertex,
		         const reco::Track &ghostTrack,
		         const reco::BeamSpot &beamSpot,
		         const std::vector<TransientTrack> &primaries,
		         const std::vector<TransientTrack> &tracks,
	                 const std::vector<float> &weights = std::vector<float>()) const;

	std::vector<TransientVertex>
		vertices(const GlobalPoint &primaryPosition,
		         const GlobalError &primaryError,
		         const GhostTrack &ghostTrack) const;

	std::vector<TransientVertex>
		vertices(const GlobalPoint &primaryPosition,
		         const GlobalError &primaryError,
		         const reco::BeamSpot &beamSpot,
		         const GhostTrack &ghostTrack) const;

	std::vector<TransientVertex>
		vertices(const GlobalPoint &primaryPosition,
		         const GlobalError &primaryError,
		         const reco::BeamSpot &beamSpot,
		         const std::vector<TransientTrack> &primaries,
		         const GhostTrack &ghostTrack) const;

	std::vector<TransientVertex>
		vertices(const reco::Vertex &primaryVertex,
		         const GhostTrack &ghostTrack) const;

	std::vector<TransientVertex>
		vertices(const reco::Vertex &primaryVertex,
		         const reco::BeamSpot &beamSpot,
		         const GhostTrack &ghostTrack) const;

	std::vector<TransientVertex>
		vertices(const reco::Vertex &primaryVertex,
		         const reco::BeamSpot &beamSpot,
		         const std::vector<TransientTrack> &primaries,
		         const GhostTrack &ghostTrack) const;

	std::vector<TransientVertex> vertices(
		const GhostTrack &ghostTrack,
		const CachingVertex<5> &primary = CachingVertex<5>(),
		const reco::BeamSpot &beamSpot = reco::BeamSpot(),
		bool hasBeamSpot = false, bool hasPrimaries = false) const;

    private:
	struct FinderInfo;

	std::vector<CachingVertex<5> > initialVertices(
						const FinderInfo &info) const;

	CachingVertex<5> mergeVertices(const CachingVertex<5> &vertex1,
	                               const CachingVertex<5> &vertex2,
	                               const FinderInfo &info,
	                               bool isPrimary) const;

	bool recursiveMerge(std::vector<CachingVertex<5> > &vertices,
	                    const FinderInfo &info) const;

	bool reassignTracks(std::vector<CachingVertex<5> > &vertices,
	                    const FinderInfo &info) const;

	void refitGhostTrack(std::vector<CachingVertex<5> > &vertices,
	                     FinderInfo &info) const;

	GhostTrackFitter &ghostTrackFitter() const;
	VertexFitter<5> &vertexFitter(bool primary) const;	

	static double vertexCompat(const CachingVertex<5> &vtx1,      
	                           const CachingVertex<5> &vtx2,
	                           const FinderInfo &info,
	                           double scale1 = 1.0, double scale2 = 1.0);

	double	maxFitChi2_;
	double	mergeThreshold_;
	double	primcut_;
	double	seccut_;
	FitType	fitType_;

	mutable std::unique_ptr<GhostTrackFitter>	ghostTrackFitter_;
	mutable std::unique_ptr<VertexFitter<5> > primVertexFitter_;
	mutable std::unique_ptr<VertexFitter<5> > secVertexFitter_;
};

}
#endif // RecoBTag_GhostTrackVertexFinder_h
