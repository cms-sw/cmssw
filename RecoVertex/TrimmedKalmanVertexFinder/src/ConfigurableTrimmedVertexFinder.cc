#include "RecoVertex/TrimmedKalmanVertexFinder/interface/ConfigurableTrimmedVertexFinder.h"
#include "CommonTools/Statistics/interface/ChiSquaredProbability.h"

using namespace reco;

ConfigurableTrimmedVertexFinder::ConfigurableTrimmedVertexFinder(const VertexFitter<5>* vf,
                                                                 const VertexUpdator<5>* vu,
                                                                 const VertexTrackCompatibilityEstimator<5>* ve)
    : theClusterFinder(vf, vu, ve),
      theVtxFitProbCut(0.01),
      theTrackCompatibilityToPV(0.05),
      theTrackCompatibilityToSV(0.01),
      theMaxNbOfVertices(0) {
  // default pt cut is 1.5 GeV
  theFilter.setPtCut(1.5);
}

void ConfigurableTrimmedVertexFinder::setParameters(const edm::ParameterSet& s) {
  theFilter.setPtCut(s.getParameter<double>("ptCut"));
  theTrackCompatibilityToPV = s.getParameter<double>("trackCompatibilityToPVcut");
  theTrackCompatibilityToSV = s.getParameter<double>("trackCompatibilityToSVcut");
  theVtxFitProbCut = s.getParameter<double>("vtxFitProbCut");
  theMaxNbOfVertices = s.getParameter<int>("maxNbOfVertices");
}

std::vector<TransientVertex> ConfigurableTrimmedVertexFinder::vertices(const std::vector<TransientTrack>& tracks) const {
  std::vector<TransientTrack> remaining;

  return vertices(tracks, remaining, reco::BeamSpot(), false);
}

std::vector<TransientVertex> ConfigurableTrimmedVertexFinder::vertices(const std::vector<TransientTrack>& tracks,
                                                                       const reco::BeamSpot& spot) const {
  std::vector<TransientTrack> remaining;
  return vertices(tracks, remaining, spot, true);
}

std::vector<TransientVertex> ConfigurableTrimmedVertexFinder::vertices(const std::vector<TransientTrack>& tracks,
                                                                       std::vector<TransientTrack>& unused,
                                                                       const reco::BeamSpot& spot,
                                                                       bool use_spot) const {
  resetEvent(tracks);
  analyseInputTracks(tracks);

  std::vector<TransientTrack> filtered;
  for (std::vector<TransientTrack>::const_iterator it = tracks.begin(); it != tracks.end(); it++) {
    if (theFilter(*it)) {
      filtered.push_back(*it);
    } else {
      unused.push_back(*it);
    }
  }

  std::vector<TransientVertex> all = vertexCandidates(filtered, unused, spot, use_spot);

  analyseVertexCandidates(all);

  std::vector<TransientVertex> sel = clean(all);

  analyseFoundVertices(sel);

  return sel;
}

std::vector<TransientVertex> ConfigurableTrimmedVertexFinder::vertexCandidates(const std::vector<TransientTrack>& tracks,
                                                                               std::vector<TransientTrack>& unused,
                                                                               const reco::BeamSpot& spot,
                                                                               bool use_spot) const {
  std::vector<TransientVertex> cand;

  std::vector<TransientTrack> remain = tracks;

  while (true) {
    float tkCompCut = (cand.empty() ? theTrackCompatibilityToPV : theTrackCompatibilityToSV);

    //    std::cout << "PVR:compat cut " << tkCompCut << std::endl;
    theClusterFinder.setTrackCompatibilityCut(tkCompCut);
    //    std::cout << "PVCF:compat cut after setting "
    //	 << theClusterFinder.trackCompatibilityCut() << std::endl;

    std::vector<TransientVertex> newVertices;
    if (cand.empty() && use_spot) {
      newVertices = theClusterFinder.vertices(remain, spot);
    } else {
      newVertices = theClusterFinder.vertices(remain);
    }
    if (newVertices.empty())
      break;

    analyseClusterFinder(newVertices, remain);

    for (std::vector<TransientVertex>::const_iterator iv = newVertices.begin(); iv != newVertices.end(); iv++) {
      if (iv->originalTracks().size() > 1) {
        cand.push_back(*iv);
      } else {
        // candidate has too few tracks - get them back into the vector
        for (std::vector<TransientTrack>::const_iterator trk = iv->originalTracks().begin();
             trk != iv->originalTracks().end();
             ++trk) {
          unused.push_back(*trk);
        }
      }
    }

    // when max number of vertices reached, stop
    if (theMaxNbOfVertices != 0) {
      if (cand.size() >= (unsigned int)theMaxNbOfVertices)
        break;
    }
  }

  for (std::vector<TransientTrack>::const_iterator it = remain.begin(); it != remain.end(); it++) {
    unused.push_back(*it);
  }

  return cand;
}

std::vector<TransientVertex> ConfigurableTrimmedVertexFinder::clean(
    const std::vector<TransientVertex>& candidates) const {
  std::vector<TransientVertex> sel;
  for (std::vector<TransientVertex>::const_iterator i = candidates.begin(); i != candidates.end(); i++) {
    if (ChiSquaredProbability((*i).totalChiSquared(), (*i).degreesOfFreedom()) > theVtxFitProbCut) {
      sel.push_back(*i);
    }
  }

  return sel;
}
