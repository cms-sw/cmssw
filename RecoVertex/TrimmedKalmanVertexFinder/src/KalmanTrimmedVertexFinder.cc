#include "RecoVertex/TrimmedKalmanVertexFinder/interface/KalmanTrimmedVertexFinder.h"
#include "RecoVertex/KalmanVertexFit/interface/KalmanVertexFitter.h"
#include "RecoVertex/KalmanVertexFit/interface/KalmanVertexUpdator.h"
#include "RecoVertex/KalmanVertexFit/interface/KalmanVertexTrackCompatibilityEstimator.h"

KalmanTrimmedVertexFinder::KalmanTrimmedVertexFinder() {
  KalmanVertexFitter vf(false);
  KalmanVertexUpdator<5> vu;
  KalmanVertexTrackCompatibilityEstimator<5> ve;

  theFinder = new ConfigurableTrimmedVertexFinder(&vf, &vu, &ve);
}

void KalmanTrimmedVertexFinder::setParameters(const edm::ParameterSet& s) {
  setPtCut(s.getParameter<double>("ptCut"));
  setTrackCompatibilityCut(s.getParameter<double>("trackCompatibilityToPVcut"));
  setTrackCompatibilityToSV(s.getParameter<double>("trackCompatibilityToSVcut"));
  setVertexFitProbabilityCut(s.getParameter<double>("vtxFitProbCut"));
  setMaxNbOfVertices(s.getParameter<int>("maxNbOfVertices"));
}

KalmanTrimmedVertexFinder::~KalmanTrimmedVertexFinder() { delete theFinder; }

KalmanTrimmedVertexFinder::KalmanTrimmedVertexFinder(const KalmanTrimmedVertexFinder& other) {
  theFinder = other.theFinder->clone();
}
