#include "RecoVertex/AdaptiveVertexFit/interface/AdaptiveVertexFitter.h"
#include "DataFormats/GeometryCommonDetAlgo/interface/GlobalError.h"
#include "RecoVertex/VertexTools/interface/AnnealingSchedule.h"
#include "RecoVertex/VertexTools/interface/GeometricAnnealing.h"
#include "RecoVertex/VertexTools/interface/VertexTrackFactory.h"
#include "RecoVertex/VertexPrimitives/interface/VertexException.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <algorithm>

using namespace edm;

// #define STORE_WEIGHTS
#ifdef STORE_WEIGHTS
#include <dataharvester/Writer.h>
#endif

using namespace std;

namespace {
  void sortTracksByPt(std::vector<reco::TransientTrack>& cont) {
    auto s = cont.size();
    float pt2[s];
    int ind[s];
    int i = 0;
    for (auto const& tk : cont) {
      ind[i] = i;
      pt2[i++] = tk.impactPointState().globalMomentum().perp2();
    }
    //clang can not handle lambdas with variable length arrays
    auto* p_pt2 = pt2;
    std::sort(ind, ind + s, [p_pt2](int i, int j) { return p_pt2[i] > p_pt2[j]; });
    std::vector<reco::TransientTrack> tmp;
    tmp.reserve(s);
    for (auto i = 0U; i < s; ++i)
      tmp.emplace_back(std::move(cont[ind[i]]));
    cont.swap(tmp);
  }

  // typedef ReferenceCountingPointer<VertexTrack<5> > RefCountedVertexTrack;
  typedef AdaptiveVertexFitter::RefCountedVertexTrack RefCountedVertexTrack;

  AlgebraicSymMatrix33 initFitError() {
    // that's how we model the lin pt error for the initial seed!
    const float initialError = 10000;
    AlgebraicSymMatrix33 ret;
    ret(0, 0) = initialError;
    ret(1, 1) = initialError;
    ret(2, 2) = initialError;
    return ret;
  }

  GlobalError const fitError = initFitError();

  AlgebraicSymMatrix33 initLinePointError() {
    // that's how we model the error of the linearization point.
    // for track weighting!
    AlgebraicSymMatrix33 ret;
    ret(0, 0) = .3;
    ret(1, 1) = .3;
    ret(2, 2) = 3.;
    // ret(0,0)=1e-7; ret(1,1)=1e-7; ret(2,2)=1e-7;
    return ret;
  }

  GlobalError const linPointError = initLinePointError();

  void sortByDistanceToRefPoint(std::vector<RefCountedVertexTrack>& cont, const GlobalPoint ref) {
    auto s = cont.size();
    float d2[s];
    int ind[s];
    int i = 0;
    for (auto const& tk : cont) {
      ind[i] = i;
      d2[i++] = (tk->linearizedTrack()->track().initialFreeState().position() - ref).mag2();
    }
    //clang can not handle lambdas with variable length arrays
    auto* p_d2 = d2;
    std::sort(ind, ind + s, [p_d2](int i, int j) { return p_d2[i] < p_d2[j]; });
    std::vector<RefCountedVertexTrack> tmp;
    tmp.reserve(s);
    for (auto i = 0U; i < s; ++i)
      tmp.emplace_back(std::move(cont[ind[i]]));
    cont.swap(tmp);
  }

#ifdef STORE_WEIGHTS
  //NOTE: This is not thread safe
  map<RefCountedLinearizedTrackState, int> ids;
  int iter = 0;

  int getId(const RefCountedLinearizedTrackState& r) {
    static int ctr = 1;
    if (ids.count(r) == 0) {
      ids[r] = ctr++;
    }
    return ids[r];
  }
#endif
}  // namespace

AdaptiveVertexFitter::AdaptiveVertexFitter(const AnnealingSchedule& ann,
                                           const LinearizationPointFinder& linP,
                                           const VertexUpdator<5>& updator,
                                           const VertexTrackCompatibilityEstimator<5>& crit,
                                           const VertexSmoother<5>& smoother,
                                           const AbstractLTSFactory<5>& ltsf)
    : theNr(0),
      theLinP(linP.clone()),
      theUpdator(updator.clone()),
      theSmoother(smoother.clone()),
      theAssProbComputer(ann.clone()),
      theComp(crit.clone()),
      theLinTrkFactory(ltsf.clone()),
      gsfIntermediarySmoothing_(false) {
  setParameters();
}

void AdaptiveVertexFitter::setWeightThreshold(float w) { theWeightThreshold = w; }

AdaptiveVertexFitter::AdaptiveVertexFitter(const AdaptiveVertexFitter& o)
    : theMaxShift(o.theMaxShift),
      theMaxLPShift(o.theMaxLPShift),
      theMaxStep(o.theMaxStep),
      theWeightThreshold(o.theWeightThreshold),
      theNr(o.theNr),
      theLinP(o.theLinP->clone()),
      theUpdator(o.theUpdator->clone()),
      theSmoother(o.theSmoother->clone()),
      theAssProbComputer(o.theAssProbComputer->clone()),
      theComp(o.theComp->clone()),
      theLinTrkFactory(o.theLinTrkFactory->clone()),
      gsfIntermediarySmoothing_(o.gsfIntermediarySmoothing_) {}

AdaptiveVertexFitter::~AdaptiveVertexFitter() {
  delete theLinP;
  delete theUpdator;
  delete theSmoother;
  delete theAssProbComputer;
  delete theComp;
  delete theLinTrkFactory;
}

void AdaptiveVertexFitter::setParameters(double maxshift, double maxlpshift, unsigned maxstep, double weightthreshold) {
  theMaxShift = maxshift;
  theMaxLPShift = maxlpshift;
  theMaxStep = maxstep;
  theWeightThreshold = weightthreshold;
}

void AdaptiveVertexFitter::setParameters(const edm::ParameterSet& s) {
  setParameters(s.getParameter<double>("maxshift"),
                s.getParameter<double>("maxlpshift"),
                s.getParameter<int>("maxstep"),
                s.getParameter<double>("weightthreshold"));
}

CachingVertex<5> AdaptiveVertexFitter::vertex(const vector<reco::TransientTrack>& unstracks) const {
  if (unstracks.size() < 2) {
    LogError("RecoVertex|AdaptiveVertexFitter") << "Supplied fewer than two tracks. Vertex is invalid.";
    return CachingVertex<5>();  // return invalid vertex
  };
  vector<reco::TransientTrack> tracks = unstracks;
  sortTracksByPt(tracks);
  // Linearization Point
  GlobalPoint linP = theLinP->getLinearizationPoint(tracks);
  // Initial vertex seed, with a very large error matrix
  VertexState lseed(linP, linPointError);
  vector<RefCountedVertexTrack> vtContainer = linearizeTracks(tracks, lseed);

  VertexState seed(linP, fitError);
  return fit(vtContainer, seed, false);
}

CachingVertex<5> AdaptiveVertexFitter::vertex(const vector<RefCountedVertexTrack>& tracks) const {
  if (tracks.size() < 2) {
    LogError("RecoVertex|AdaptiveVertexFitter") << "Supplied fewer than two tracks. Vertex is invalid.";
    return CachingVertex<5>();  // return invalid vertex
  };
  // Initial vertex seed, with a very small weight matrix
  GlobalPoint linP = tracks[0]->linearizedTrack()->linearizationPoint();
  VertexState seed(linP, fitError);
  return fit(tracks, seed, false);
}

CachingVertex<5> AdaptiveVertexFitter::vertex(const vector<RefCountedVertexTrack>& tracks,
                                              const reco::BeamSpot& spot) const {
  if (tracks.empty()) {
    LogError("RecoVertex|AdaptiveVertexFitter") << "Supplied no tracks. Vertex is invalid.";
    return CachingVertex<5>();  // return invalid vertex
  };
  VertexState beamSpotState(spot);
  return fit(tracks, beamSpotState, true);
}

/** Fit vertex out of a set of reco::TransientTracks.
 *  Uses the specified linearization point.
 */
CachingVertex<5> AdaptiveVertexFitter::vertex(const vector<reco::TransientTrack>& tracks,
                                              const GlobalPoint& linPoint) const {
  if (tracks.size() < 2) {
    LogError("RecoVertex|AdaptiveVertexFitter") << "Supplied fewer than two tracks. Vertex is invalid.";
    return CachingVertex<5>();  // return invalid vertex
  };
  // Initial vertex seed, with a very large error matrix
  VertexState seed(linPoint, linPointError);
  vector<RefCountedVertexTrack> vtContainer = linearizeTracks(tracks, seed);
  VertexState fitseed(linPoint, fitError);
  return fit(vtContainer, fitseed, false);
}

/** Fit vertex out of a set of TransientTracks. 
 *  The specified BeamSpot will be used as priot, but NOT for the linearization.
 *  The specified LinearizationPointFinder will be used to find the linearization point.
 */
CachingVertex<5> AdaptiveVertexFitter::vertex(const vector<reco::TransientTrack>& unstracks,
                                              const reco::BeamSpot& beamSpot) const {
  if (unstracks.empty()) {
    LogError("RecoVertex|AdaptiveVertexFitter") << "Supplied no tracks. Vertex is invalid.";
    return CachingVertex<5>();  // return invalid vertex
  };

  VertexState beamSpotState(beamSpot);
  vector<RefCountedVertexTrack> vtContainer;

  vector<reco::TransientTrack> tracks = unstracks;
  sortTracksByPt(tracks);

  if (tracks.size() > 1) {
    // Linearization Point search if there are more than 1 track
    GlobalPoint linP = theLinP->getLinearizationPoint(tracks);
    VertexState lpState(linP, linPointError);
    vtContainer = linearizeTracks(tracks, lpState);
  } else {
    // otherwise take the beamspot position.
    vtContainer = linearizeTracks(tracks, beamSpotState);
  }

  return fit(vtContainer, beamSpotState, true);
}

/** Fit vertex out of a set of reco::TransientTracks.
 *   Uses the position as both the linearization point AND as prior
 *   estimate of the vertex position. The error is used for the
 *   weight of the prior estimate.
 */
CachingVertex<5> AdaptiveVertexFitter::vertex(const vector<reco::TransientTrack>& tracks,
                                              const GlobalPoint& priorPos,
                                              const GlobalError& priorError) const

{
  if (tracks.empty()) {
    LogError("RecoVertex|AdaptiveVertexFitter") << "Supplied no tracks. Vertex is invalid.";
    return CachingVertex<5>();  // return invalid vertex
  };
  VertexState seed(priorPos, priorError);
  vector<RefCountedVertexTrack> vtContainer = linearizeTracks(tracks, seed);
  return fit(vtContainer, seed, true);
}

/** Fit vertex out of a set of VertexTracks
 *   Uses the position and error for the prior estimate of the vertex.
 *   This position is not used to relinearize the tracks.
 */
CachingVertex<5> AdaptiveVertexFitter::vertex(const vector<RefCountedVertexTrack>& tracks,
                                              const GlobalPoint& priorPos,
                                              const GlobalError& priorError) const {
  if (tracks.empty()) {
    LogError("RecoVertex|AdaptiveVertexFitter") << "Supplied no tracks. Vertex is invalid.";
    return CachingVertex<5>();  // return invalid vertex
  };
  VertexState seed(priorPos, priorError);
  return fit(tracks, seed, true);
}

/**
 * Construct a container of VertexTrack from a set of reco::TransientTracks.
 * As this is the first iteration of the adaptive fit, the initial error
 * does not enter in the computation of the weights.
 * This is to avoid that all tracks get the same weight when
 * using a very large initial error matrix.
 */
vector<AdaptiveVertexFitter::RefCountedVertexTrack> AdaptiveVertexFitter::linearizeTracks(
    const vector<reco::TransientTrack>& tracks, const VertexState& seed) const {
  const GlobalPoint& linP(seed.position());
  vector<RefCountedLinearizedTrackState> lTracks;
  for (vector<reco::TransientTrack>::const_iterator i = tracks.begin(); i != tracks.end(); ++i) {
    try {
      RefCountedLinearizedTrackState lTrData = theLinTrkFactory->linearizedTrackState(linP, *i);
      lTracks.push_back(lTrData);
    } catch (exception& e) {
      LogInfo("RecoVertex/AdaptiveVertexFitter") << "Exception " << e.what() << " in ::linearizeTracks."
                                                 << "Your future vertex has just lost a track.";
    };
  }
  return weightTracks(lTracks, seed);
}

/**
 * Construct new a container of VertexTrack with a new linearization point
 * and vertex seed, from an existing set of VertexTrack, from which only the
 * recTracks will be used.
 */
vector<AdaptiveVertexFitter::RefCountedVertexTrack> AdaptiveVertexFitter::reLinearizeTracks(
    const vector<RefCountedVertexTrack>& tracks, const CachingVertex<5>& vertex) const {
  const VertexState& seed = vertex.vertexState();
  GlobalPoint linP = seed.position();
  vector<RefCountedLinearizedTrackState> lTracks;
  for (vector<RefCountedVertexTrack>::const_iterator i = tracks.begin(); i != tracks.end(); i++) {
    try {
      RefCountedLinearizedTrackState lTrData =
          theLinTrkFactory->linearizedTrackState(linP, (**i).linearizedTrack()->track());
      /*
      RefCountedLinearizedTrackState lTrData =
              (**i).linearizedTrack()->stateWithNewLinearizationPoint(linP);
              */
      lTracks.push_back(lTrData);
    } catch (exception& e) {
      LogInfo("RecoVertex/AdaptiveVertexFitter") << "Exception " << e.what() << " in ::relinearizeTracks. "
                                                 << "Will not relinearize this track.";
      lTracks.push_back((**i).linearizedTrack());
    };
  };
  return reWeightTracks(lTracks, vertex);
}

AdaptiveVertexFitter* AdaptiveVertexFitter::clone() const { return new AdaptiveVertexFitter(*this); }

double AdaptiveVertexFitter::getWeight(float chi2) const {
  double weight = theAssProbComputer->weight(chi2);

  if (weight > 1.0) {
    LogInfo("RecoVertex/AdaptiveVertexFitter") << "Weight " << weight << " > 1.0!";
    weight = 1.0;
  };

  if (weight < 1e-20) {
    // LogInfo("RecoVertex/AdaptiveVertexFitter") << "Weight " << weight << " < 0.0!";
    weight = 1e-20;
  };
  return weight;
}

vector<AdaptiveVertexFitter::RefCountedVertexTrack> AdaptiveVertexFitter::reWeightTracks(
    const vector<RefCountedLinearizedTrackState>& lTracks, const CachingVertex<5>& vertex) const {
  const VertexState& seed = vertex.vertexState();
  // cout << "[AdaptiveVertexFitter] now reweight around " << seed.position() << endl;
  theNr++;
  // GlobalPoint pos = seed.position();

  vector<RefCountedVertexTrack> finalTracks;
  VertexTrackFactory<5> vTrackFactory;
#ifdef STORE_WEIGHTS
  iter++;
#endif
  for (vector<RefCountedLinearizedTrackState>::const_iterator i = lTracks.begin(); i != lTracks.end(); i++) {
    double weight = 0.;
    // cout << "[AdaptiveVertexFitter] estimate " << endl;
    pair<bool, double> chi2Res(false, 0.);
    try {
      chi2Res = theComp->estimate(vertex, *i, std::distance(lTracks.begin(), i));
    } catch (exception const& e) {
    };
    // cout << "[AdaptiveVertexFitter] /estimate " << endl;
    if (!chi2Res.first) {
      // cout << "[AdaptiveVertexFitter] aie... vertex candidate is at  " << vertex.position() << endl;
      LogInfo("AdaptiveVertexFitter") << "When reweighting, chi2<0. Will add this track with w=0.";
      // edm::LogInfo("AdaptiveVertexFitter" ) << "pt=" << (**i).track().pt();
    } else {
      weight = getWeight(chi2Res.second);
    }

    RefCountedVertexTrack vTrData = vTrackFactory.vertexTrack(*i, seed, weight);

#ifdef STORE_WEIGHTS
    map<string, dataharvester::MultiType> m;
    m["chi2"] = chi2;
    m["w"] = theAssProbComputer->weight(chi2);
    m["T"] = theAssProbComputer->currentTemp();
    m["n"] = iter;
    m["pos"] = "reweight";
    m["id"] = getId(*i);
    dataharvester::Writer::file("w.txt").save(m);
#endif

    finalTracks.push_back(vTrData);
  }
  sortByDistanceToRefPoint(finalTracks, vertex.position());
  // cout << "[AdaptiveVertexFitter] /now reweight" << endl;
  return finalTracks;
}

vector<AdaptiveVertexFitter::RefCountedVertexTrack> AdaptiveVertexFitter::weightTracks(
    const vector<RefCountedLinearizedTrackState>& lTracks, const VertexState& seed) const {
  theNr++;
  CachingVertex<5> seedvtx(seed, vector<RefCountedVertexTrack>(), 0.);
  /** track weighting, as opposed to re-weighting, must always 
   * be done with a reset annealer! */
  theAssProbComputer->resetAnnealing();

  vector<RefCountedVertexTrack> finalTracks;
  VertexTrackFactory<5> vTrackFactory;
#ifdef STORE_WEIGHTS
  iter++;
#endif
  for (vector<RefCountedLinearizedTrackState>::const_iterator i = lTracks.begin(); i != lTracks.end(); i++) {
    double weight = 0.;
    pair<bool, double> chi2Res = theComp->estimate(seedvtx, *i, std::distance(lTracks.begin(), i));
    if (!chi2Res.first) {
      // cout << "[AdaptiveVertexFitter] Aiee! " << endl;
      LogInfo("AdaptiveVertexFitter") << "When weighting a track, chi2 calculation failed;"
                                      << " will add with w=0.";
    } else {
      weight = getWeight(chi2Res.second);
    }
    RefCountedVertexTrack vTrData = vTrackFactory.vertexTrack(*i, seed, weight);
#ifdef STORE_WEIGHTS
    map<string, dataharvester::MultiType> m;
    m["chi2"] = chi2;
    m["w"] = theAssProbComputer->weight(chi2);
    m["T"] = theAssProbComputer->currentTemp();
    m["n"] = iter;
    m["id"] = getId(*i);
    m["pos"] = "weight";
    dataharvester::Writer::file("w.txt").save(m);
#endif
    finalTracks.push_back(vTrData);
  }
  return finalTracks;
}

/**
 * Construct new a container of VertexTrack with new weights
 * accounting for vertex error, from an existing set of VertexTracks.
 * From these the LinearizedTracks will be reused.
 */
vector<AdaptiveVertexFitter::RefCountedVertexTrack> AdaptiveVertexFitter::reWeightTracks(
    const vector<RefCountedVertexTrack>& tracks, const CachingVertex<5>& seed) const {
  vector<RefCountedLinearizedTrackState> lTracks;
  for (vector<RefCountedVertexTrack>::const_iterator i = tracks.begin(); i != tracks.end(); i++) {
    lTracks.push_back((**i).linearizedTrack());
  }

  return reWeightTracks(lTracks, seed);
}

/*
 * The method where the vertex fit is actually done!
 */

CachingVertex<5> AdaptiveVertexFitter::fit(const vector<RefCountedVertexTrack>& tracks,
                                           const VertexState& priorSeed,
                                           bool withPrior) const {
  // cout << "[AdaptiveVertexFit] fit with " << tracks.size() << endl;
  theAssProbComputer->resetAnnealing();

  vector<RefCountedVertexTrack> initialTracks;
  GlobalPoint priorVertexPosition = priorSeed.position();
  GlobalError priorVertexError = priorSeed.error();

  CachingVertex<5> returnVertex(priorVertexPosition, priorVertexError, initialTracks, 0);
  if (withPrior) {
    returnVertex = CachingVertex<5>(
        priorVertexPosition, priorVertexError, priorVertexPosition, priorVertexError, initialTracks, 0);
  }

  std::vector<RefCountedVertexTrack> globalVTracks = tracks;
  // sort the tracks, according to distance to seed!
  sortByDistanceToRefPoint(globalVTracks, priorSeed.position());

  // main loop through all the VTracks
  int step = 0;

  CachingVertex<5> initialVertex = returnVertex;

  GlobalPoint newPosition = priorVertexPosition;
  GlobalPoint previousPosition = newPosition;

  int ns_trks = 0;  // number of significant tracks.
  // If we have only two significant tracks, we return an invalid vertex

  // cout << "[AdaptiveVertexFit] start " << tracks.size() << endl;
  /*
  for ( vector< RefCountedVertexTrack >::const_iterator 
        i=globalVTracks.begin(); i!=globalVTracks.end() ; ++i )
  {
    cout << "  " << (**i).linearizedTrack()->track().initialFreeState().momentum() << endl;
  }*/
  do {
    ns_trks = 0;
    CachingVertex<5> fVertex = initialVertex;
    // cout << "[AdaptiveVertexFit] step " << step << " at " << fVertex.position() << endl;
    if ((previousPosition - newPosition).transverse() > theMaxLPShift) {
      // relinearize and reweight.
      // (reLinearizeTracks also reweights tracks)
      // cout << "[AdaptiveVertexFit] relinearize at " << returnVertex.position() << endl;
      if (gsfIntermediarySmoothing_)
        returnVertex = theSmoother->smooth(returnVertex);
      globalVTracks = reLinearizeTracks(globalVTracks, returnVertex);
    } else if (step) {
      // reweight, if it is not the first step
      // cout << "[AdaptiveVertexFit] reweight at " << returnVertex.position() << endl;
      if (gsfIntermediarySmoothing_)
        returnVertex = theSmoother->smooth(returnVertex);
      globalVTracks = reWeightTracks(globalVTracks, returnVertex);
    }
    // cout << "[AdaptiveVertexFit] relinarized, reweighted" << endl;
    // update sequentially the vertex estimate
    CachingVertex<5> nVertex;
    for (vector<RefCountedVertexTrack>::const_iterator i = globalVTracks.begin(); i != globalVTracks.end(); i++) {
      if ((**i).weight() > 0.)
        nVertex = theUpdator->add(fVertex, *i);
      else
        nVertex = fVertex;
      if (nVertex.isValid()) {
        if ((**i).weight() >= theWeightThreshold)
          ns_trks++;

        if (fabs(nVertex.position().z()) > 10000. || nVertex.position().perp() > 120.) {
          // were more than 100 m off!!
          LogInfo("AdaptiveVertexFitter")
              << "Vertex candidate just took off to " << nVertex.position() << "! Will discard this update!";
          // 	    //<< "track pt was " << (**i).linearizedTrack()->track().pt()
          // 					     << "track momentum was " << (**i).linearizedTrack()->track().initialFreeState().momentum()
          // 					     << "track position was " << (**i).linearizedTrack()->track().initialFreeState().position()
          // 					     << "track chi2 was " << (**i).linearizedTrack()->track().chi2()
          // 					     << "track ndof was " << (**i).linearizedTrack()->track().ndof()
          // 					     << "track w was " << (**i).weight()
          // 					     << "track schi2 was " << (**i).smoothedChi2();
        } else {
          fVertex = nVertex;
        }
      } else {
        LogInfo("RecoVertex/AdaptiveVertexFitter")
            << "The updator returned an invalid vertex when adding track " << i - globalVTracks.begin()
            << ".\n Your vertex might just have lost one good track.";
      }
    }
    previousPosition = newPosition;
    newPosition = fVertex.position();
    returnVertex = fVertex;
    theAssProbComputer->anneal();
    step++;
    if (step >= theMaxStep)
      break;

  } while (
      // repeat as long as
      // - vertex moved too much or
      // - we're not yet annealed
      (((previousPosition - newPosition).mag() > theMaxShift) || (!(theAssProbComputer->isAnnealed()))));

  if (theWeightThreshold > 0. && ns_trks < 2 && !withPrior) {
    LogDebug("AdaptiveVertexFitter") << "fewer than two significant tracks (w>" << theWeightThreshold << ")."
                                     << " Fitted vertex is invalid.";
    return CachingVertex<5>();  // return invalid vertex
  }

#ifdef STORE_WEIGHTS
  map<string, dataharvester::MultiType> m;
  m["chi2"] = chi2;
  m["w"] = theAssProbComputer->weight(chi2);
  m["T"] = theAssProbComputer->currentTemp();
  m["n"] = iter;
  m["id"] = getId(*i);
  m["pos"] = "final";
  dataharvester::Writer::file("w.txt").save(m);
#endif
  // cout << "[AdaptiveVertexFit] /fit" << endl;
  return theSmoother->smooth(returnVertex);
}
