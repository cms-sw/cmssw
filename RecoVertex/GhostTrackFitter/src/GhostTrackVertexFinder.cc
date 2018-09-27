#include <algorithm>
#include <iterator>
#include <vector>
#include <map>
#include <set>

#include <Math/SVector.h>
#include <Math/SMatrix.h>
#include <Math/MatrixFunctions.h>

#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "DataFormats/GeometryVector/interface/GlobalVector.h"
#include "DataFormats/GeometryCommonDetAlgo/interface/GlobalError.h"

#include "DataFormats/BeamSpot/interface/BeamSpot.h"

#include "TrackingTools/TrajectoryParametrization/interface/GlobalTrajectoryParameters.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
#include "TrackingTools/TrajectoryState/interface/FreeTrajectoryState.h"
#include "TrackingTools/GeomPropagators/interface/AnalyticalImpactPointExtrapolator.h"
#include "TrackingTools/TransientTrack/interface/TransientTrack.h"
#include "TrackingTools/TransientTrack/interface/TransientTrackFromFTSFactory.h"

#include "RecoVertex/VertexPrimitives/interface/TransientVertex.h"
#include "RecoVertex/VertexPrimitives/interface/VertexFitter.h"
#include "RecoVertex/VertexPrimitives/interface/VertexState.h"
#include "RecoVertex/VertexPrimitives/interface/CachingVertex.h"
#include "RecoVertex/VertexPrimitives/interface/LinearizedTrackState.h"
#include "RecoVertex/VertexPrimitives/interface/ConvertError.h"
#include "RecoVertex/VertexPrimitives/interface/ConvertToFromReco.h"
#include "RecoVertex/VertexTools/interface/LinearizedTrackStateFactory.h"
#include "RecoVertex/VertexTools/interface/VertexTrackFactory.h"
#include "RecoVertex/VertexTools/interface/VertexDistance3D.h"
#include "RecoVertex/VertexTools/interface/GeometricAnnealing.h"
#include "RecoVertex/KalmanVertexFit/interface/KalmanVertexFitter.h"
#include "RecoVertex/AdaptiveVertexFit/interface/AdaptiveVertexFitter.h"

#include "RecoVertex/GhostTrackFitter/interface/GhostTrack.h"
#include "RecoVertex/GhostTrackFitter/interface/GhostTrackState.h"
#include "RecoVertex/GhostTrackFitter/interface/GhostTrackFitter.h"
#include "RecoVertex/GhostTrackFitter/interface/GhostTrackPrediction.h"
#include "RecoVertex/GhostTrackFitter/interface/SequentialGhostTrackFitter.h"
#include "RecoVertex/GhostTrackFitter/interface/KalmanGhostTrackUpdater.h"

#include "RecoVertex/GhostTrackFitter/interface/GhostTrackVertexFinder.h"

// #define DEBUG

#ifdef DEBUG
#	include "RecoBTag/SecondaryVertex/interface/TrackKinematics.h"
#endif

using namespace reco;

namespace {
	using namespace ROOT::Math;

	typedef SVector<double, 3> Vector3;

	typedef SMatrix<double, 3, 3, MatRepSym<double, 3> > Matrix3S;
	typedef SMatrix<double, 2, 2, MatRepSym<double, 2> > Matrix2S;
	typedef SMatrix<double, 2, 3> Matrix23;

	typedef ReferenceCountingPointer<VertexTrack<5> >
					RefCountedVertexTrack;
	typedef ReferenceCountingPointer<LinearizedTrackState<5> >
					RefCountedLinearizedTrackState;

	struct VtxTrackIs : public std::unary_function<
					RefCountedVertexTrack, bool> {
		VtxTrackIs(const TransientTrack &track) : track(track) {}
		bool operator()(const RefCountedVertexTrack &vtxTrack) const
		{ return vtxTrack->linearizedTrack()->track() == track; }

		const TransientTrack &track;
	};

	inline Vector3 conv(const GlobalVector &vec)
	{
		Vector3 result;
		result[0] = vec.x();
		result[1] = vec.y();
		result[2] = vec.z();
		return result;
	}

	inline double sqr(double arg) { return arg * arg; }
}

struct GhostTrackVertexFinder::FinderInfo {
	FinderInfo(const CachingVertex<5> &primaryVertex,
	           const GhostTrack &ghostTrack, const BeamSpot &beamSpot,
	           bool hasBeamSpot, bool hasPrimaries) :
		primaryVertex(primaryVertex), pred(ghostTrack.prediction()),
		prior(ghostTrack.prior()), states(ghostTrack.states()),
		beamSpot(beamSpot), hasBeamSpot(hasBeamSpot),
		hasPrimaries(hasPrimaries), field(nullptr) {}

	const CachingVertex<5>			&primaryVertex;
	GhostTrackPrediction			pred;
	GhostTrackPrediction			prior;
	std::vector<GhostTrackState>		states;
	VertexState				beamSpot;
	bool					hasBeamSpot;
	bool					hasPrimaries;
	const MagneticField			*field;
	TransientTrack				ghostTrack;
};

static TransientTrack transientGhostTrack(const GhostTrackPrediction &pred,
                                          const MagneticField *field)
{ return TransientTrackFromFTSFactory().build(pred.fts(field)); }

static double vtxErrorLong(const GlobalError &error, const GlobalVector &dir)
{
	return ROOT::Math::Similarity(conv(dir), error.matrix());
}

static GlobalPoint vtxMean(const GlobalPoint &p1, const GlobalError &e1,
                           const GlobalPoint &p2, const GlobalError &e2)
{
	GlobalVector diff = p2 - p1;

	double err1 = vtxErrorLong(e1, diff);
	double err2 = vtxErrorLong(e2, diff);

	double weight = err1 / (err1 + err2);

	return p1 + weight * diff;
}

static VertexState stateMean(const VertexState &v1, const VertexState &v2)
{
	return VertexState(vtxMean(v1.position(), v1.error(),
	                           v2.position(), v2.error()),
	                   v1.error() + v2.error());
}

static bool covarianceUpdate(Matrix3S &cov, const Vector3 &residual,
                             const Matrix3S &error, double &chi2,
                             double theta, double phi)
{
	using namespace ROOT::Math;

	Matrix23 jacobian;
	jacobian(0, 0) = std::cos(phi) * std::cos(theta);
	jacobian(0, 1) = std::sin(phi) * std::cos(theta);
	jacobian(0, 2) = -std::sin(theta);
	jacobian(1, 0) = -std::sin(phi);
	jacobian(1, 1) = std::cos(phi);

	Matrix2S measErr = Similarity(jacobian, error);
	Matrix2S combErr = Similarity(jacobian, cov) + measErr;
	if (!measErr.Invert() || !combErr.Invert())
		return false;

	cov -= SimilarityT(jacobian * cov, combErr);
	chi2 += Similarity(jacobian * residual, measErr);

	return true;
}

static CachingVertex<5> vertexAtState(const TransientTrack &ghostTrack,
                                      const GhostTrackPrediction &pred,
                                      const GhostTrackState &state)
{
	LinearizedTrackStateFactory linTrackFactory;
	VertexTrackFactory<5> vertexTrackFactory;

	if (!state.isValid())
		return CachingVertex<5>();

	GlobalPoint pca1 = pred.position(state.lambda());
	GlobalError err1 = pred.positionError(state.lambda());

	GlobalPoint pca2 = state.globalPosition();
	GlobalError err2 = state.cartesianError();

	GlobalPoint point = vtxMean(pca1, err1, pca2, err2);

	const TransientTrack& recTrack = state.track();

	RefCountedLinearizedTrackState linState[2] = {
		linTrackFactory.linearizedTrackState(point, ghostTrack),
		linTrackFactory.linearizedTrackState(point, recTrack)
	};
	if ( !linState[0]->isValid() || !linState[1]->isValid() )
 	  return CachingVertex<5>();

	Matrix3S cov = SMatrixIdentity();
	cov *= 10000;
	double chi2 = 0.;
	if (!covarianceUpdate(cov, conv(pca1 - point), err1.matrix(), chi2,
	                       linState[0]->predictedStateParameters()[1],
	                       linState[0]->predictedStateParameters()[2]) ||
	    !covarianceUpdate(cov, conv(pca2 - point), err2.matrix(), chi2,
	                       linState[1]->predictedStateParameters()[1],
	                       linState[1]->predictedStateParameters()[2]))
		return CachingVertex<5>();

	GlobalError error(cov);
	VertexState vtxState(point, error);

	std::vector<RefCountedVertexTrack> linTracks(2);
	linTracks[0] = vertexTrackFactory.vertexTrack(linState[0], vtxState);
	linTracks[1] = vertexTrackFactory.vertexTrack(linState[1], vtxState);

	return CachingVertex<5>(point, error, linTracks, chi2);
}

static RefCountedVertexTrack relinearizeTrack(
			const RefCountedVertexTrack &track,
			const VertexState &state,
			const VertexTrackFactory<5> factory)
{
	RefCountedLinearizedTrackState linTrack = track->linearizedTrack();
	linTrack = linTrack->stateWithNewLinearizationPoint(state.position());
	return factory.vertexTrack(linTrack, state);
}

static std::vector<RefCountedVertexTrack> relinearizeTracks(
			const std::vector<RefCountedVertexTrack> &tracks,
			const VertexState &state)
{
	VertexTrackFactory<5> vertexTrackFactory;

	std::vector<RefCountedVertexTrack> finalTracks;
	finalTracks.reserve(tracks.size());

	for(std::vector<RefCountedVertexTrack>::const_iterator iter =
				tracks.begin(); iter != tracks.end(); ++iter)
		finalTracks.push_back(
			relinearizeTrack(*iter, state, vertexTrackFactory));

	return finalTracks;
}

double GhostTrackVertexFinder::vertexCompat(const CachingVertex<5> &vtx1,
                           const CachingVertex<5> &vtx2,
                           const FinderInfo &info,
                           double scale1, double scale2)
{
	Vector3 diff = conv(vtx2.position() - vtx1.position());
	Matrix3S cov = scale1 * vtx1.error().matrix() +
	               scale2 * vtx2.error().matrix();

	return sqr(ROOT::Math::Dot(diff, diff)) / ROOT::Math::Similarity(cov, diff);
}

static double trackVertexCompat(const CachingVertex<5> &vtx,
                                const RefCountedVertexTrack &vertexTrack)
{
	using namespace ROOT::Math;

	TransientTrack track = vertexTrack->linearizedTrack()->track();
	GlobalPoint point = vtx.position();
	AnalyticalImpactPointExtrapolator extrap(track.field());
	TrajectoryStateOnSurface tsos =
			extrap.extrapolate(track.impactPointState(), point);

	if (!tsos.isValid())
		return 1.0e6;

	GlobalPoint point1 = vtx.position();
	GlobalPoint point2 = tsos.globalPosition();
	Vector3 dir = conv(point2 - point1);
	Matrix3S error = vtx.error().matrix() +
	                 tsos.cartesianError().matrix().Sub<Matrix3S>(0, 0);
	if (!error.Invert())
		return 1.0e6;

	return ROOT::Math::Similarity(error, dir);
}

GhostTrackVertexFinder::GhostTrackVertexFinder() :
	maxFitChi2_(10.0),
	mergeThreshold_(3.0),
	primcut_(2.0),
	seccut_(5.0),
	fitType_(kAlwaysWithGhostTrack)
{
}

GhostTrackVertexFinder::GhostTrackVertexFinder(
		double maxFitChi2, double mergeThreshold, double primcut,
		double seccut, FitType fitType) :
	maxFitChi2_(maxFitChi2),
	mergeThreshold_(mergeThreshold),
	primcut_(primcut),
	seccut_(seccut),
	fitType_(fitType)
{
}

GhostTrackVertexFinder::~GhostTrackVertexFinder()
{
}

GhostTrackFitter &GhostTrackVertexFinder::ghostTrackFitter() const
{
	if (!ghostTrackFitter_.get())
		ghostTrackFitter_.reset(new GhostTrackFitter);

	return *ghostTrackFitter_;
}

VertexFitter<5> &GhostTrackVertexFinder::vertexFitter(bool primary) const
{
	std::unique_ptr<VertexFitter<5> > *ptr =
			primary ? &primVertexFitter_ : &secVertexFitter_;
	if (!ptr->get())
		ptr->reset(new AdaptiveVertexFitter(
					GeometricAnnealing(primary
						? primcut_ : seccut_)));

	return **ptr;
}

#ifdef DEBUG
static void debugTrack(const TransientTrack &track, const TransientVertex *vtx = 0)
{
	const FreeTrajectoryState &fts = track.initialFreeState();
	GlobalVector mom = fts.momentum();
	std::cout << "\tpt = " << mom.perp() << ", eta = " << mom.eta() << ", phi = " << mom.phi() << ", charge = " << fts.charge();
	if (vtx && vtx->hasTrackWeight())
		std::cout << ", w = " << vtx->trackWeight(track) << std::endl;
	else
		std::cout << std::endl;
}

static void debugTracks(const std::vector<TransientTrack> &tracks, const TransientVertex *vtx = 0)
{
	TrackKinematics kin;
	for(std::vector<TransientTrack>::const_iterator iter = tracks.begin();
	    iter != tracks.end(); ++iter) {
		debugTrack(*iter, vtx);
		try {
			TrackBaseRef track = iter->trackBaseRef();
			kin.add(*track);
		} catch ( exception const & e ) {
			// ignore, yeah this is debugging, so shut up ^^
		}
	}

	std::cout << "mass = " << kin.vectorSum().M() << std::endl;
}

static void debugTracks(const std::vector<RefCountedVertexTrack> &tracks)
{
	std::vector<TransientTrack> tracks_;
	for(std::vector<RefCountedVertexTrack>::const_iterator iter = tracks.begin();
	    iter != tracks.end(); ++iter) {
	    	std::cout << "+ " << (*iter)->linearizedTrack()->linearizationPoint() << std::endl;
		tracks_.push_back((*iter)->linearizedTrack()->track());
	}
	debugTracks(tracks_);
}

static void debugVertex(const TransientVertex &vtx, const GhostTrackPrediction &pred)
{
	double origin = 0.;
	std::cout << vtx.position() << "-> " << vtx.totalChiSquared() << " with " << vtx.degreesOfFreedom() << std::endl;

	double err = std::sqrt(vtxErrorLong(
			vtx.positionError(),
			pred.direction())
					/ pred.rho2());
	std::cout << "* " << (pred.lambda(vtx.position()) * pred.rho() - origin)
	          << " +-" << err
	          << " => " << ((pred.lambda(vtx.position()) * pred.rho() - origin) / err)
	          << std::endl;

	std::vector<TransientTrack> tracks = vtx.originalTracks();
	debugTracks(tracks, &vtx);
}
#endif

std::vector<TransientVertex> GhostTrackVertexFinder::vertices(
			const Vertex &primaryVertex,
			const GlobalVector &direction,
			double coneRadius,
			const std::vector<TransientTrack> &tracks) const
{
	return vertices(RecoVertex::convertPos(primaryVertex.position()),
	                RecoVertex::convertError(primaryVertex.error()),
	                direction, coneRadius, tracks);
}

std::vector<TransientVertex> GhostTrackVertexFinder::vertices(
			const Vertex &primaryVertex,
			const GlobalVector &direction,
			double coneRadius,
			const reco::BeamSpot &beamSpot,
			const std::vector<TransientTrack> &tracks) const
{
	return vertices(RecoVertex::convertPos(primaryVertex.position()),
	                RecoVertex::convertError(primaryVertex.error()),
	                direction, coneRadius, beamSpot, tracks);
}

std::vector<TransientVertex> GhostTrackVertexFinder::vertices(
			const Vertex &primaryVertex,
			const GlobalVector &direction,
			double coneRadius,
			const reco::BeamSpot &beamSpot,
			const std::vector<TransientTrack> &primaries,
			const std::vector<TransientTrack> &tracks) const
{
	return vertices(RecoVertex::convertPos(primaryVertex.position()),
	                RecoVertex::convertError(primaryVertex.error()),
	                direction, coneRadius, beamSpot, primaries, tracks);
}

std::vector<TransientVertex> GhostTrackVertexFinder::vertices(
			const GlobalPoint &primaryPosition,
			const GlobalError &primaryError,
			const GlobalVector &direction,
			double coneRadius,
			const std::vector<TransientTrack> &tracks) const
{
	GhostTrack ghostTrack =
		ghostTrackFitter().fit(primaryPosition, primaryError,
		                       direction, coneRadius, tracks);

	CachingVertex<5> primary(primaryPosition, primaryError,
	                         std::vector<RefCountedVertexTrack>(), 0.);

	std::vector<TransientVertex> result = vertices(ghostTrack, primary);

#ifdef DEBUG
	for(std::vector<TransientVertex>::const_iterator iter = result.begin();
	    iter != result.end(); ++iter)
		debugVertex(*iter, ghostTrack.prediction());
#endif

	return result;
}

std::vector<TransientVertex> GhostTrackVertexFinder::vertices(
			const GlobalPoint &primaryPosition,
			const GlobalError &primaryError,
			const GlobalVector &direction,
			double coneRadius,
			const BeamSpot &beamSpot,
			const std::vector<TransientTrack> &tracks) const
{
	GhostTrack ghostTrack =
		ghostTrackFitter().fit(primaryPosition, primaryError,
		                       direction, coneRadius, tracks);

	CachingVertex<5> primary(primaryPosition, primaryError,
	                         std::vector<RefCountedVertexTrack>(), 0.);

	std::vector<TransientVertex> result =
				vertices(ghostTrack, primary, beamSpot, true);

#ifdef DEBUG
	for(std::vector<TransientVertex>::const_iterator iter = result.begin();
	    iter != result.end(); ++iter)
		debugVertex(*iter, ghostTrack.prediction());
#endif

	return result;
}

std::vector<TransientVertex> GhostTrackVertexFinder::vertices(
			const GlobalPoint &primaryPosition,
			const GlobalError &primaryError,
			const GlobalVector &direction,
			double coneRadius,
			const BeamSpot &beamSpot,
			const std::vector<TransientTrack> &primaries,
			const std::vector<TransientTrack> &tracks) const
{
	GhostTrack ghostTrack =
		ghostTrackFitter().fit(primaryPosition, primaryError,
		                       direction, coneRadius, tracks);

	std::vector<RefCountedVertexTrack> primaryVertexTracks;
	if (!primaries.empty()) {
		LinearizedTrackStateFactory linTrackFactory;
		VertexTrackFactory<5> vertexTrackFactory;

		VertexState state(primaryPosition, primaryError);

		for(std::vector<TransientTrack>::const_iterator iter =
			primaries.begin(); iter != primaries.end(); ++iter) {

			RefCountedLinearizedTrackState linState =
				linTrackFactory.linearizedTrackState(
						primaryPosition, *iter);

			primaryVertexTracks.push_back(
					vertexTrackFactory.vertexTrack(
							linState, state));
		}
	}

	CachingVertex<5> primary(primaryPosition, primaryError,
	                         primaryVertexTracks, 0.);

	std::vector<TransientVertex> result =
			vertices(ghostTrack, primary, beamSpot, true, true);

#ifdef DEBUG
	for(std::vector<TransientVertex>::const_iterator iter = result.begin();
	    iter != result.end(); ++iter)
		debugVertex(*iter, ghostTrack.prediction());
#endif

	return result;
}

std::vector<TransientVertex> GhostTrackVertexFinder::vertices(
			const GlobalPoint &primaryPosition,
			const GlobalError &primaryError,
			const GhostTrack &ghostTrack) const
{
	CachingVertex<5> primary(primaryPosition, primaryError,
	                         std::vector<RefCountedVertexTrack>(), 0.);

	std::vector<TransientVertex> result = vertices(ghostTrack, primary);

#ifdef DEBUG
	for(std::vector<TransientVertex>::const_iterator iter = result.begin();
	    iter != result.end(); ++iter)
		debugVertex(*iter, ghostTrack.prediction());
#endif

	return result;
}


std::vector<TransientVertex> GhostTrackVertexFinder::vertices(
			const GlobalPoint &primaryPosition,
			const GlobalError &primaryError,
			const BeamSpot &beamSpot,
			const GhostTrack &ghostTrack) const
{
	CachingVertex<5> primary(primaryPosition, primaryError,
	                         std::vector<RefCountedVertexTrack>(), 0.);

	std::vector<TransientVertex> result =
				vertices(ghostTrack, primary, beamSpot, true);

#ifdef DEBUG
	for(std::vector<TransientVertex>::const_iterator iter = result.begin();
	    iter != result.end(); ++iter)
		debugVertex(*iter, ghostTrack.prediction());
#endif

	return result;
}

std::vector<TransientVertex> GhostTrackVertexFinder::vertices(
			const GlobalPoint &primaryPosition,
			const GlobalError &primaryError,
			const BeamSpot &beamSpot,
			const std::vector<TransientTrack> &primaries,
			const GhostTrack &ghostTrack) const
{
	std::vector<RefCountedVertexTrack> primaryVertexTracks;
	if (!primaries.empty()) {
		LinearizedTrackStateFactory linTrackFactory;
		VertexTrackFactory<5> vertexTrackFactory;

		VertexState state(primaryPosition, primaryError);

		for(std::vector<TransientTrack>::const_iterator iter =
			primaries.begin(); iter != primaries.end(); ++iter) {

			RefCountedLinearizedTrackState linState =
				linTrackFactory.linearizedTrackState(
						primaryPosition, *iter);

			primaryVertexTracks.push_back(
					vertexTrackFactory.vertexTrack(
							linState, state));
		}
	}

	CachingVertex<5> primary(primaryPosition, primaryError,
	                         primaryVertexTracks, 0.);

	std::vector<TransientVertex> result =
			vertices(ghostTrack, primary, beamSpot, true, true);

#ifdef DEBUG
	for(std::vector<TransientVertex>::const_iterator iter = result.begin();
	    iter != result.end(); ++iter)
		debugVertex(*iter, ghostTrack.prediction());
#endif

	return result;
}

std::vector<TransientVertex> GhostTrackVertexFinder::vertices(
			const Vertex &primaryVertex,
			const GhostTrack &ghostTrack) const
{
	return vertices(RecoVertex::convertPos(primaryVertex.position()),
	                RecoVertex::convertError(primaryVertex.error()),
	                ghostTrack);
}

std::vector<TransientVertex> GhostTrackVertexFinder::vertices(
			const Vertex &primaryVertex,
			const BeamSpot &beamSpot,
			const GhostTrack &ghostTrack) const
{
	return vertices(RecoVertex::convertPos(primaryVertex.position()),
	                RecoVertex::convertError(primaryVertex.error()),
	                beamSpot, ghostTrack);
}

std::vector<TransientVertex> GhostTrackVertexFinder::vertices(
			const Vertex &primaryVertex,
			const BeamSpot &beamSpot,
			const std::vector<TransientTrack> &primaries,
			const GhostTrack &ghostTrack) const
{
	return vertices(RecoVertex::convertPos(primaryVertex.position()),
	                RecoVertex::convertError(primaryVertex.error()),
	                beamSpot, primaries, ghostTrack);
}

static GhostTrackPrediction dummyPrediction(
		const Vertex &primaryVertex, const Track &ghostTrack)
{
	return GhostTrackPrediction(
			RecoVertex::convertPos(primaryVertex.position()),
			RecoVertex::convertError(primaryVertex.error()),
			GlobalVector(ghostTrack.px(), ghostTrack.py(),
			             ghostTrack.pz()), 0.05);
}

std::vector<TransientVertex> GhostTrackVertexFinder::vertices(
			const Vertex &primaryVertex,
			const Track &ghostTrack,
			const std::vector<TransientTrack> &tracks,
                        const std::vector<float> &weights) const
{
	GhostTrack ghostTrack_(ghostTrack, tracks, weights,
	                       dummyPrediction(primaryVertex, ghostTrack),
	                       RecoVertex::convertPos(
	                       			primaryVertex.position()),
	                       true);

	CachingVertex<5> primary(
			RecoVertex::convertPos(primaryVertex.position()),
	                RecoVertex::convertError(primaryVertex.error()),
			std::vector<RefCountedVertexTrack>(), 0.);

	std::vector<TransientVertex> result = vertices(ghostTrack_, primary);

#ifdef DEBUG
	for(std::vector<TransientVertex>::const_iterator iter = result.begin();
	    iter != result.end(); ++iter)
		debugVertex(*iter, ghostTrack_.prediction());
#endif

	return result;
}

std::vector<TransientVertex> GhostTrackVertexFinder::vertices(
			const Vertex &primaryVertex,
			const Track &ghostTrack,
			const BeamSpot &beamSpot,
			const std::vector<TransientTrack> &tracks,
                        const std::vector<float> &weights) const
{
	GhostTrack ghostTrack_(ghostTrack, tracks, weights,
	                       dummyPrediction(primaryVertex, ghostTrack),
	                       RecoVertex::convertPos(primaryVertex.position()),
	                       true);

	CachingVertex<5> primary(
			RecoVertex::convertPos(primaryVertex.position()),
	                RecoVertex::convertError(primaryVertex.error()),
			std::vector<RefCountedVertexTrack>(), 0.);

	std::vector<TransientVertex> result =
			vertices(ghostTrack_, primary, beamSpot, true);

#ifdef DEBUG
	for(std::vector<TransientVertex>::const_iterator iter = result.begin();
	    iter != result.end(); ++iter)
		debugVertex(*iter, ghostTrack_.prediction());
#endif

	return result;
}

std::vector<TransientVertex> GhostTrackVertexFinder::vertices(
			const Vertex &primaryVertex,
			const Track &ghostTrack,
			const BeamSpot &beamSpot,
			const std::vector<TransientTrack> &primaries,
			const std::vector<TransientTrack> &tracks,
                        const std::vector<float> &weights) const
{
	GhostTrack ghostTrack_(ghostTrack, tracks, weights,
	                       dummyPrediction(primaryVertex, ghostTrack),
	                       RecoVertex::convertPos(primaryVertex.position()),
	                       true);

	std::vector<RefCountedVertexTrack> primaryVertexTracks;
	if (!primaries.empty()) {
		LinearizedTrackStateFactory linTrackFactory;
		VertexTrackFactory<5> vertexTrackFactory;

		VertexState state(
		        RecoVertex::convertPos(primaryVertex.position()),
	                RecoVertex::convertError(primaryVertex.error()));

		for(std::vector<TransientTrack>::const_iterator iter =
			primaries.begin(); iter != primaries.end(); ++iter) {

			RefCountedLinearizedTrackState linState =
				linTrackFactory.linearizedTrackState(
						state.position(), *iter);

			primaryVertexTracks.push_back(
					vertexTrackFactory.vertexTrack(
							linState, state));
		}
	}

	CachingVertex<5> primary(
			RecoVertex::convertPos(primaryVertex.position()),
	                RecoVertex::convertError(primaryVertex.error()),
			primaryVertexTracks, 0.);

	std::vector<TransientVertex> result =
			vertices(ghostTrack_, primary, beamSpot, true, true);

#ifdef DEBUG
	for(std::vector<TransientVertex>::const_iterator iter = result.begin();
	    iter != result.end(); ++iter)
		debugVertex(*iter, ghostTrack_.prediction());
#endif

	return result;
}

static double fitChi2(const CachingVertex<5> &vtx)
{
	return vtx.totalChiSquared() / vtx.degreesOfFreedom();
}

std::vector<CachingVertex<5> > GhostTrackVertexFinder::initialVertices(
						const FinderInfo &info) const
{
	std::vector<CachingVertex<5> > vertices;
	for(std::vector<GhostTrackState>::const_iterator iter =
		    info.states.begin(); iter != info. states.end(); ++iter) {

		if (!iter->isValid())
			continue;

		GhostTrackState state(*iter);
		state.linearize(info.pred);

		if (!state.isValid())
			continue;

		CachingVertex<5> vtx = vertexAtState(info.ghostTrack,
		                                     info.pred, state);

		if (vtx.isValid()) // && fitChi2(vtx) < maxFitChi2_)
			vertices.push_back(vtx);
	}

	return vertices;
}

static void mergeTrackHelper(const std::vector<RefCountedVertexTrack> &tracks,
                             std::vector<RefCountedVertexTrack> &newTracks,
                             const VertexState &state,
                             const VtxTrackIs &ghostTrackFinder,
                             RefCountedVertexTrack &ghostTrack,
                             const VertexTrackFactory<5> &factory)
{
	for(std::vector<RefCountedVertexTrack>::const_iterator iter =
			tracks.begin(); iter != tracks.end(); ++iter) {
		bool gt = ghostTrackFinder(*iter);
		if (gt && ghostTrack)
			continue;

		RefCountedVertexTrack track =
				relinearizeTrack(*iter, state, factory);

		if (gt)
			ghostTrack = *iter;
		else
			newTracks.push_back(*iter);
	}
}

CachingVertex<5> GhostTrackVertexFinder::mergeVertices(
					const CachingVertex<5> &vertex1,
					const CachingVertex<5> &vertex2,
					const FinderInfo &info,
					bool isPrimary) const
{
	VertexTrackFactory<5> vertexTrackFactory;

	VertexState state = stateMean(vertex1.vertexState(),
	                              vertex2.vertexState());
	std::vector<RefCountedVertexTrack> linTracks;
	VtxTrackIs isGhostTrack(info.ghostTrack);
	RefCountedVertexTrack vtxGhostTrack;

	mergeTrackHelper(vertex1.tracks(), linTracks, state,
	                 isGhostTrack, vtxGhostTrack, vertexTrackFactory);
	mergeTrackHelper(vertex2.tracks(), linTracks, state,
	                 isGhostTrack, vtxGhostTrack, vertexTrackFactory);

	if (vtxGhostTrack &&
	    (fitType_ == kAlwaysWithGhostTrack || linTracks.size() < 2))
		linTracks.push_back(
			vertexTrackFactory.vertexTrack(
				vtxGhostTrack->linearizedTrack(),
				vtxGhostTrack->vertexState()));

	try {
		CachingVertex<5> vtx;
		if (info.hasBeamSpot && isPrimary)
			vtx = vertexFitter(true).vertex(linTracks,
	                                	info.beamSpot.position(),
						info.beamSpot.error());
		else
			vtx = vertexFitter(isPrimary).vertex(linTracks);
		if (vtx.isValid())
			return vtx;
	} catch(const VertexException &e) {
		// fit failed
	}

	return CachingVertex<5>();
}

bool GhostTrackVertexFinder::recursiveMerge(
				std::vector<CachingVertex<5> > &vertices,
				const FinderInfo &info) const
{
	typedef std::pair<unsigned int, unsigned int> Indices;

	std::multimap<float, Indices> compatMap;
	unsigned int n = vertices.size();
	for(unsigned int i = 0; i < n; i++) {
		const CachingVertex<5> &v1 = vertices[i];
		for(unsigned int j = i + 1; j < n; j++) {
			const CachingVertex<5> &v2 = vertices[j];

			float compat = vertexCompat(v1, v2, info,
					i == 0 ? primcut_ : seccut_, seccut_);

			if (compat > mergeThreshold_)
				continue;

			compatMap.insert(
				std::make_pair(compat, Indices(i, j)));
		}
	}

	bool changesMade = false;
	bool repeat = true;
	while(repeat) {
		repeat = false;
		for(std::multimap<float, Indices>::const_iterator iter =
			compatMap.begin(); iter != compatMap.end(); ++iter) {
			unsigned int v1 = iter->second.first;
			unsigned int v2 = iter->second.second;

			CachingVertex<5> newVtx =
				mergeVertices(vertices[v1], vertices[v2],
				              info, v1 == 0);
			if (!newVtx.isValid() ||
			    (v1 != 0 && fitChi2(newVtx) > maxFitChi2_))
				continue;

			std::swap(vertices[v1], newVtx);
			vertices.erase(vertices.begin() + v2);
			n--;

			std::multimap<float, Indices> newCompatMap;
			for(++iter; iter != compatMap.end(); ++iter) {
				if (iter->second.first == v1 ||
				    iter->second.first == v2 ||
				    iter->second.second == v1 ||
				    iter->second.second == v2)
					continue;

				Indices indices = iter->second;
				indices.first -= indices.first > v2;
				indices.second -= indices.second > v2;

				newCompatMap.insert(std::make_pair(
							iter->first, indices));
			}
			std::swap(compatMap, newCompatMap);

			for(unsigned int i = 0; i < n; i++) {
				if (i == v1)
					continue;

				const CachingVertex<5> &other = vertices[i];
				float compat = vertexCompat(
						vertices[v1], other, info,
						v1 == 0 ? primcut_ : seccut_,
						i == 0 ? primcut_ : seccut_);

				if (compat > mergeThreshold_)
					continue;

				compatMap.insert(
					std::make_pair(
						compat,
						Indices(std::min(i, v1),
						        std::max(i, v1))));
			}

			changesMade = true;
			repeat = true;
			break;
		}
	}

	return changesMade;
}

bool GhostTrackVertexFinder::reassignTracks(
				std::vector<CachingVertex<5> > &vertices_,
				const FinderInfo &info) const
{
	std::vector<std::pair<RefCountedVertexTrack,
	                      std::vector<RefCountedVertexTrack> > >
					trackBundles(vertices_.size());

	VtxTrackIs isGhostTrack(info.ghostTrack);

	bool assignmentChanged = false;
	for(std::vector<CachingVertex<5> >::const_iterator iter =
			vertices_.begin(); iter != vertices_.end(); ++iter) {
		std::vector<RefCountedVertexTrack> vtxTracks = iter->tracks();

		if (vtxTracks.empty()) {
			LinearizedTrackStateFactory linTrackFactory;
			VertexTrackFactory<5> vertexTrackFactory;

			RefCountedLinearizedTrackState linState =
				linTrackFactory.linearizedTrackState(
					iter->position(), info.ghostTrack);

			trackBundles[iter - vertices_.begin()].first =
					vertexTrackFactory.vertexTrack(
						linState, iter->vertexState());
		}

		for(std::vector<RefCountedVertexTrack>::const_iterator track =
			vtxTracks.begin(); track != vtxTracks.end(); ++track) {

			if (isGhostTrack(*track)) {
				trackBundles[iter - vertices_.begin()]
							.first = *track;
				continue;
			}

			if ((*track)->weight() < 1e-3) {
				trackBundles[iter - vertices_.begin()]
						.second.push_back(*track);
				continue;
			}

			unsigned int idx = iter - vertices_.begin();
			double best = 1.0e9;
			for(std::vector<CachingVertex<5> >::const_iterator
						vtx = vertices_.begin();
			    vtx != vertices_.end(); ++vtx) {
				if (info.pred.lambda(vtx->position()) <
				    info.pred.lambda(vertices_[0].position()))
					continue;

				double compat =
					sqr(trackVertexCompat(*vtx, *track));

				compat /= (vtx == vertices_.begin()) ?
							primcut_ : seccut_;

				if (compat < best) {
					best = compat;
					idx = vtx - vertices_.begin();
				}
			}

			if ((int)idx != iter - vertices_.begin())
				assignmentChanged = true;

			trackBundles[idx].second.push_back(*track);
		}
	}

	if (!assignmentChanged)
		return false;

	VertexTrackFactory<5> vertexTrackFactory;
	std::vector<CachingVertex<5> > vertices;
	vertices.reserve(vertices_.size());

	for(std::vector<CachingVertex<5> >::const_iterator iter =
			vertices_.begin(); iter != vertices_.end(); ++iter) {
		const std::vector<RefCountedVertexTrack> &tracks =
				trackBundles[iter - vertices_.begin()].second;
		if (tracks.empty())
			continue;

		CachingVertex<5> vtx;

		if (tracks.size() == 1) {
			const TransientTrack &track =
				tracks[0]->linearizedTrack()->track();

			int idx = -1;
			for(std::vector<GhostTrackState>::const_iterator iter =
							info.states.begin();
			    iter != info.states.end(); ++iter) {
				if (iter->isTrack() && iter->track() == track) {
					idx = iter - info.states.begin();
					break;
				}
			}
			if (idx < 0)
				continue;

			vtx = vertexAtState(info.ghostTrack, info.pred,
			                    info.states[idx]);
			if (!vtx.isValid())
				continue;
		} else {
			std::vector<RefCountedVertexTrack> linTracks =
				relinearizeTracks(tracks, iter->vertexState());

			if (fitType_ == kAlwaysWithGhostTrack)
				linTracks.push_back(relinearizeTrack(
					trackBundles[iter - vertices_.begin()].first,
					iter->vertexState(), vertexTrackFactory));

			bool primary = iter == vertices_.begin();
			try {
				if (primary && info.hasBeamSpot)
					vtx = vertexFitter(true).vertex(
						linTracks,
						info.beamSpot.position(),
						info.beamSpot.error());
				else
					vtx = vertexFitter(primary).vertex(
								linTracks);
			} catch(const VertexException &e) {
				// fit failed;
			}
			if (!vtx.isValid())
				return false;
		}

		vertices.push_back(vtx);
	};

	std::swap(vertices_, vertices);
	return true;
}

void GhostTrackVertexFinder::refitGhostTrack(
				std::vector<CachingVertex<5> > &vertices,
				FinderInfo &info) const
{
	VtxTrackIs isGhostTrack(info.ghostTrack);

	std::vector<GhostTrackState> states;
	std::vector<unsigned int> oldStates;
	oldStates.reserve(info.states.size());

	for(std::vector<CachingVertex<5> >::const_iterator iter =
			vertices.begin(); iter != vertices.end(); ++iter) {
		std::vector<RefCountedVertexTrack> vtxTracks = iter->tracks();

		oldStates.clear();
		for(std::vector<RefCountedVertexTrack>::const_iterator track =
			vtxTracks.begin(); track != vtxTracks.end(); ++track) {

			if (isGhostTrack(*track) || (*track)->weight() < 1e-3)
				continue;

			const TransientTrack &tt =
					(*track)->linearizedTrack()->track();

			int idx = -1;
			for(std::vector<GhostTrackState>::const_iterator iter =
							info.states.begin();
			    iter != info.states.end(); ++iter) {
				if (iter->isTrack() && iter->track() == tt) {
					idx = iter - info.states.begin();
					break;
				}
			}

			if (idx >= 0)
				oldStates.push_back(idx);
		}

		if (oldStates.size() == 1)
			states.push_back(info.states[oldStates[0]]);
		else
			states.push_back(GhostTrackState(iter->vertexState()));
	}

	KalmanGhostTrackUpdater updater;
	SequentialGhostTrackFitter fitter;
	double ndof, chi2;
	info.pred = fitter.fit(updater, info.prior, states, ndof, chi2);
	TransientTrack ghostTrack = transientGhostTrack(info.pred, info.field);

	std::swap(info.states, states);
	states.clear();

	std::vector<CachingVertex<5> > newVertices;
	for(std::vector<CachingVertex<5> >::const_iterator iter =
			vertices.begin(); iter != vertices.end(); ++iter) {
		std::vector<RefCountedVertexTrack> vtxTracks = iter->tracks();

		int idx = -1;
		bool redo = false;
		for(std::vector<RefCountedVertexTrack>::iterator track =
			vtxTracks.begin(); track != vtxTracks.end(); ++track) {

			if (isGhostTrack(*track)) {
				LinearizedTrackStateFactory linTrackFactory;
				VertexTrackFactory<5> vertexTrackFactory;

				*track = vertexTrackFactory.vertexTrack(
					linTrackFactory.linearizedTrackState(
						iter->position(), ghostTrack),
						iter->vertexState());
				redo = true;
				continue;
			}

			const TransientTrack &tt =
					(*track)->linearizedTrack()->track();

			if (idx >= 0) {
				idx = -1;
				break;
			}

			for(std::vector<GhostTrackState>::const_iterator it =
							info.states.begin();
			    it != info.states.end(); ++it) {
				if (!it->isTrack())
					continue;

				if (it->track() == tt) {
					idx = it - info.states.begin();
					break;
				}
			}
		}

		if (idx >= 0) {
			CachingVertex<5> vtx =
				vertexAtState(ghostTrack, info.pred,
				              info.states[idx]);
			if (vtx.isValid())
				newVertices.push_back(vtx);
		} else if (redo) {
			bool primary = iter == vertices.begin();
			CachingVertex<5> vtx;
			if (primary && info.hasBeamSpot)
				vtx = vertexFitter(true).vertex(
						vtxTracks,
						info.beamSpot.position(),
						info.beamSpot.error());
			else
				vtx = vertexFitter(primary).vertex(vtxTracks);
			if (vtx.isValid())
				newVertices.push_back(vtx);
		} else
			newVertices.push_back(*iter);
	}

	std::swap(newVertices, vertices);
	info.ghostTrack = ghostTrack;
}

// implementation

std::vector<TransientVertex> GhostTrackVertexFinder::vertices(
				const GhostTrack &ghostTrack,
				const CachingVertex<5> &primary,
				const BeamSpot &beamSpot,
				bool hasBeamSpot, bool hasPrimaries) const
{
	FinderInfo info(primary, ghostTrack, beamSpot,
	                hasBeamSpot, hasPrimaries);

	std::vector<TransientVertex> result;
	if (info.states.empty())
		return result;

	info.field = info.states[0].track().field();
	info.ghostTrack = transientGhostTrack(info.pred, info.field);

	std::vector<CachingVertex<5> > vertices = initialVertices(info);
	if (primary.isValid()) {
		vertices.push_back(primary);
		if (vertices.size() > 1)
			std::swap(vertices.front(), vertices.back());
	}

	unsigned int reassigned = 0;
	while(reassigned < 3) {
		if (vertices.size() < 2)
			break;

#ifdef DEBUG
		for(std::vector<CachingVertex<5> >::const_iterator iter =
			vertices.begin(); iter != vertices.end(); ++iter)

			debugVertex(*iter, ghostTrack.prediction());

		std::cout << "----- recursive merging: ---------" << std::endl;
#endif

		bool changed = recursiveMerge(vertices, info);
		if ((!changed && !reassigned) || vertices.size() < 2)
			break;
		if (changed)
			reassigned = 0;

		if (fitType_ == kRefitGhostTrackWithVertices) {
			refitGhostTrack(vertices, info);
			changed = true;
		}

		try {
#ifdef DEBUG
			for(std::vector<CachingVertex<5> >::const_iterator
						iter = vertices.begin();
			    iter != vertices.end(); ++iter)
				debugVertex(*iter, ghostTrack.prediction());
			std::cout << "----- reassignment: ---------" << std::endl;
#endif
			if (reassignTracks(vertices, info)) {
				reassigned++;
				changed = true;
			} else
				reassigned = 0;
		} catch(const VertexException &e) {
			// just keep vertices as they are
		}

		if (!changed)
			break;
	}

	for(std::vector<CachingVertex<5> >::const_iterator iter =
			vertices.begin(); iter != vertices.end(); ++iter) {
		std::vector<RefCountedVertexTrack> tracks = iter->tracks();
		std::vector<RefCountedVertexTrack> newTracks;
		newTracks.reserve(tracks.size());

		std::remove_copy_if(tracks.begin(), tracks.end(),
		                    std::back_inserter(newTracks),
		                    VtxTrackIs(info.ghostTrack));

		if (newTracks.empty())
			continue;

		CachingVertex<5> vtx(iter->vertexState(), newTracks,
		                     iter->totalChiSquared());
		result.push_back(vtx);
	}

	return result;
}

