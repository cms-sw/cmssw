#include "RecoVertex/VertexPrimitives/interface/TransientVertex.h"
#include "RecoVertex/VertexPrimitives/interface/VertexException.h"
#include "TrackingTools/TransientTrack/interface/TrackTransientTrack.h"
#include "TrackingTools/TransientTrack/interface/CandidatePtrTransientTrack.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <algorithm>

using namespace std;
using namespace reco;

TransientVertex::TransientVertex()
    : theVertexState(),
      theOriginalTracks(),
      theChi2(0),
      theNDF(0),
      vertexValid(false),
      withPrior(false),
      theWeightMapIsAvailable(false),
      theCovMapAvailable(false),
      withRefittedTracks(false) {}

TransientVertex::TransientVertex(const GlobalPoint& pos,
                                 const GlobalError& posError,
                                 const std::vector<TransientTrack>& tracks,
                                 float chi2)
    : theVertexState(pos, posError),
      theOriginalTracks(tracks),
      theChi2(chi2),
      theNDF(0),
      vertexValid(true),
      withPrior(false),
      theWeightMapIsAvailable(false),
      theCovMapAvailable(false),
      withRefittedTracks(false) {
  theNDF = 2. * theOriginalTracks.size() - 3.;
}

TransientVertex::TransientVertex(const GlobalPoint& pos,
                                 const double time,
                                 const GlobalError& posTimeError,
                                 const std::vector<TransientTrack>& tracks,
                                 float chi2)
    : theVertexState(pos, time, posTimeError),
      theOriginalTracks(tracks),
      theChi2(chi2),
      theNDF(0),
      vertexValid(true),
      withPrior(false),
      theWeightMapIsAvailable(false),
      theCovMapAvailable(false),
      withRefittedTracks(false) {
  theNDF = 2. * theOriginalTracks.size() - 3.;
}

TransientVertex::TransientVertex(const GlobalPoint& pos,
                                 const GlobalError& posError,
                                 const std::vector<TransientTrack>& tracks,
                                 float chi2,
                                 float ndf)
    : theVertexState(pos, posError),
      theOriginalTracks(tracks),
      theChi2(chi2),
      theNDF(ndf),
      vertexValid(true),
      withPrior(false),
      theWeightMapIsAvailable(false),
      theCovMapAvailable(false),
      withRefittedTracks(false) {}

TransientVertex::TransientVertex(const GlobalPoint& pos,
                                 const double time,
                                 const GlobalError& posTimeError,
                                 const std::vector<TransientTrack>& tracks,
                                 float chi2,
                                 float ndf)
    : theVertexState(pos, time, posTimeError),
      theOriginalTracks(tracks),
      theChi2(chi2),
      theNDF(ndf),
      vertexValid(true),
      withPrior(false),
      theWeightMapIsAvailable(false),
      theCovMapAvailable(false),
      withRefittedTracks(false) {}

TransientVertex::TransientVertex(const GlobalPoint& priorPos,
                                 const GlobalError& priorErr,
                                 const GlobalPoint& pos,
                                 const GlobalError& posError,
                                 const std::vector<TransientTrack>& tracks,
                                 float chi2)
    : thePriorVertexState(priorPos, priorErr),
      theVertexState(pos, posError),
      theOriginalTracks(tracks),
      theChi2(chi2),
      theNDF(0),
      vertexValid(true),
      withPrior(true),
      theWeightMapIsAvailable(false),
      theCovMapAvailable(false),
      withRefittedTracks(false) {
  theNDF = 2. * theOriginalTracks.size();
}

TransientVertex::TransientVertex(const GlobalPoint& priorPos,
                                 const double priorTime,
                                 const GlobalError& priorErr,
                                 const GlobalPoint& pos,
                                 const double time,
                                 const GlobalError& posError,
                                 const std::vector<TransientTrack>& tracks,
                                 float chi2)
    : thePriorVertexState(priorPos, priorTime, priorErr),
      theVertexState(pos, time, posError),
      theOriginalTracks(tracks),
      theChi2(chi2),
      theNDF(0),
      vertexValid(true),
      withPrior(true),
      theWeightMapIsAvailable(false),
      theCovMapAvailable(false),
      withRefittedTracks(false) {
  theNDF = 2. * theOriginalTracks.size();
}

TransientVertex::TransientVertex(const GlobalPoint& priorPos,
                                 const GlobalError& priorErr,
                                 const GlobalPoint& pos,
                                 const GlobalError& posError,
                                 const std::vector<TransientTrack>& tracks,
                                 float chi2,
                                 float ndf)
    : thePriorVertexState(priorPos, priorErr),
      theVertexState(pos, posError),
      theOriginalTracks(tracks),
      theChi2(chi2),
      theNDF(ndf),
      vertexValid(true),
      withPrior(true),
      theWeightMapIsAvailable(false),
      theCovMapAvailable(false),
      withRefittedTracks(false) {}

TransientVertex::TransientVertex(const GlobalPoint& priorPos,
                                 const double priorTime,
                                 const GlobalError& priorErr,
                                 const GlobalPoint& pos,
                                 const double time,
                                 const GlobalError& posError,
                                 const std::vector<TransientTrack>& tracks,
                                 float chi2,
                                 float ndf)
    : thePriorVertexState(priorPos, priorTime, priorErr),
      theVertexState(pos, time, posError),
      theOriginalTracks(tracks),
      theChi2(chi2),
      theNDF(ndf),
      vertexValid(true),
      withPrior(true),
      theWeightMapIsAvailable(false),
      theCovMapAvailable(false),
      withRefittedTracks(false) {}

TransientVertex::TransientVertex(const VertexState& state, const std::vector<TransientTrack>& tracks, float chi2)
    : theVertexState(state),
      theOriginalTracks(tracks),
      theChi2(chi2),
      theNDF(0),
      vertexValid(true),
      withPrior(false),
      theWeightMapIsAvailable(false),
      theCovMapAvailable(false),
      withRefittedTracks(false) {
  theNDF = 2. * theOriginalTracks.size() - 3.;
}

TransientVertex::TransientVertex(const VertexState& state,
                                 const std::vector<TransientTrack>& tracks,
                                 float chi2,
                                 float ndf)
    : theVertexState(state),
      theOriginalTracks(tracks),
      theChi2(chi2),
      theNDF(ndf),
      vertexValid(true),
      withPrior(false),
      theWeightMapIsAvailable(false),
      theCovMapAvailable(false),
      withRefittedTracks(false) {}

TransientVertex::TransientVertex(const VertexState& prior,
                                 const VertexState& state,
                                 const std::vector<TransientTrack>& tracks,
                                 float chi2)
    : thePriorVertexState(prior),
      theVertexState(state),
      theOriginalTracks(tracks),
      theChi2(chi2),
      theNDF(0),
      vertexValid(true),
      withPrior(true),
      theWeightMapIsAvailable(false),
      theCovMapAvailable(false),
      withRefittedTracks(false) {
  theNDF = 2. * theOriginalTracks.size();
}

TransientVertex::TransientVertex(const VertexState& prior,
                                 const VertexState& state,
                                 const std::vector<TransientTrack>& tracks,
                                 float chi2,
                                 float ndf)
    : thePriorVertexState(prior),
      theVertexState(state),
      theOriginalTracks(tracks),
      theChi2(chi2),
      theNDF(ndf),
      vertexValid(true),
      withPrior(true),
      theWeightMapIsAvailable(false),
      theCovMapAvailable(false),
      withRefittedTracks(false) {}

void TransientVertex::weightMap(const TransientTrackToFloatMap& theMap) {
  theWeightMap = theMap;
  theWeightMapIsAvailable = true;
}

void TransientVertex::refittedTracks(const std::vector<reco::TransientTrack>& refittedTracks) {
  if (refittedTracks.empty())
    throw VertexException("TransientVertex::refittedTracks: No refitted tracks stored in input container");
  theRefittedTracks = refittedTracks;
  withRefittedTracks = true;
}

void TransientVertex::tkToTkCovariance(const TTtoTTmap& covMap) {
  theCovMap = covMap;
  theCovMapAvailable = true;
}

float TransientVertex::trackWeight(const TransientTrack& track) const {
  if (!theWeightMapIsAvailable) {
    std::vector<TransientTrack>::const_iterator foundTrack =
        find(theOriginalTracks.begin(), theOriginalTracks.end(), track);
    return ((foundTrack != theOriginalTracks.end()) ? 1. : 0.);
  }
  TransientTrackToFloatMap::const_iterator it = theWeightMap.find(track);
  if (it != theWeightMap.end()) {
    return (it->second);
  }
  return 0.;
}

AlgebraicMatrix33 TransientVertex::tkToTkCovariance(const TransientTrack& t1, const TransientTrack& t2) const {
  if (!theCovMapAvailable) {
    throw VertexException("TransientVertex::Track-to-track covariance matrices not available");
  }
  const TransientTrack* tr1;
  const TransientTrack* tr2;
  if (t1 < t2) {
    tr1 = &t1;
    tr2 = &t2;
  } else {
    tr1 = &t2;
    tr2 = &t1;
  }
  TTtoTTmap::const_iterator it = theCovMap.find(*tr1);
  if (it != theCovMap.end()) {
    const TTmap& tm = it->second;
    TTmap::const_iterator nit = tm.find(*tr2);
    if (nit != tm.end()) {
      return (nit->second);
    } else {
      throw VertexException("TransientVertex::requested Track-to-track covariance matrix does not exist");
    }
  } else {
    throw VertexException("TransientVertex::requested Track-to-track covariance matrix does not exist");
  }
}

TransientTrack TransientVertex::originalTrack(const TransientTrack& refTrack) const {
  if (theRefittedTracks.empty())
    throw VertexException("TransientVertex::requested No refitted tracks stored in vertex");
  std::vector<TransientTrack>::const_iterator it = find(theRefittedTracks.begin(), theRefittedTracks.end(), refTrack);
  if (it == theRefittedTracks.end())
    throw VertexException(
        "TransientVertex::requested Refitted track not found in list.\n address used for comparison: ")
        << refTrack.basicTransientTrack();
  size_t pos = it - theRefittedTracks.begin();
  return theOriginalTracks[pos];
}

TransientTrack TransientVertex::refittedTrack(const TransientTrack& track) const {
  if (theRefittedTracks.empty())
    throw VertexException("TransientVertex::requested No refitted tracks stored in vertex");
  std::vector<TransientTrack>::const_iterator it = find(theOriginalTracks.begin(), theOriginalTracks.end(), track);
  if (it == theOriginalTracks.end())
    throw VertexException("transientVertex::requested Track not found in list.\n address used for comparison: ")
        << track.basicTransientTrack();
  size_t pos = it - theOriginalTracks.begin();
  return theRefittedTracks[pos];
}

TransientVertex::operator reco::Vertex() const {
  //If the vertex is invalid, return an invalid TV !
  if (!isValid())
    return Vertex();

  Vertex vertex(Vertex::Point(theVertexState.position()),
                // 	RecoVertex::convertError(theVertexState.error()),
                theVertexState.error4D().matrix4D(),
                theVertexState.time(),
                totalChiSquared(),
                degreesOfFreedom(),
                theOriginalTracks.size());
  for (std::vector<TransientTrack>::const_iterator i = theOriginalTracks.begin(); i != theOriginalTracks.end(); ++i) {
    //     const TrackTransientTrack* ttt = dynamic_cast<const TrackTransientTrack*>((*i).basicTransientTrack());
    //     if ((ttt!=0) && (ttt->persistentTrackRef().isNonnull()))
    //     {
    //       TrackRef tr = ttt->persistentTrackRef();
    //       TrackBaseRef tbr(tr);
    if (withRefittedTracks) {
      vertex.add((*i).trackBaseRef(), refittedTrack(*i).track(), trackWeight(*i));
    } else {
      vertex.add((*i).trackBaseRef(), trackWeight(*i));
    }
    //}
  }
  return vertex;
}

TransientVertex::operator reco::VertexCompositePtrCandidate() const {
  using namespace reco;
  if (!isValid())
    return VertexCompositePtrCandidate();

  VertexCompositePtrCandidate vtxCompPtrCand;

  vtxCompPtrCand.setTime(vertexState().time());
  vtxCompPtrCand.setCovariance(vertexState().error4D().matrix4D());
  vtxCompPtrCand.setChi2AndNdof(totalChiSquared(), degreesOfFreedom());
  vtxCompPtrCand.setVertex(Candidate::Point(position().x(), position().y(), position().z()));

  Candidate::LorentzVector p4;
  for (std::vector<reco::TransientTrack>::const_iterator tt = theOriginalTracks.begin(); tt != theOriginalTracks.end();
       ++tt) {
    if (trackWeight(*tt) < 0.5)
      continue;

    const CandidatePtrTransientTrack* cptt = dynamic_cast<const CandidatePtrTransientTrack*>(tt->basicTransientTrack());
    if (cptt == nullptr)
      edm::LogError("DynamicCastingFailed") << "Casting of TransientTrack to CandidatePtrTransientTrack failed!";
    else {
      p4 += cptt->candidate()->p4();
      vtxCompPtrCand.addDaughter(cptt->candidate());
    }
  }

  //TODO: if has refitted tracks we should scale the candidate p4 to the refitted one
  vtxCompPtrCand.setP4(p4);
  return vtxCompPtrCand;
}
