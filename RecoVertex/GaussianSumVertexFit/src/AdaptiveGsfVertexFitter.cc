#include "RecoVertex/GaussianSumVertexFit/interface/AdaptiveGsfVertexFitter.h"
#include "RecoVertex/VertexTools/interface/DummyVertexSmoother.h"
#include "RecoVertex/GaussianSumVertexFit/interface/GsfVertexSmoother.h"
#include "RecoVertex/GaussianSumVertexFit/interface/MultiPerigeeLTSFactory.h"
#include "RecoVertex/GaussianSumVertexFit/interface/GsfVertexTrackCompatibilityEstimator.h"
#include "RecoVertex/VertexTools/interface/GeometricAnnealing.h"



AdaptiveGsfVertexFitter::AdaptiveGsfVertexFitter(const edm::ParameterSet& pSet,
	const LinearizationPointFinder & linP )
{
  float theMaxShift = pSet.getParameter<double>("maxDistance"); //0.01
  int theMaxStep = pSet.getParameter<int>("maxNbrOfIterations"); //10
  bool limitComponents_ = pSet.getParameter<bool>("limitComponents");
  bool useSmoothing = pSet.getParameter<bool>("smoothTracks");

  DeepCopyPointerByClone<GsfVertexMerger> theMerger;

  if (limitComponents_) {
    edm::ParameterSet mergerPSet = pSet.getParameter<edm::ParameterSet>("GsfMergerParameters");
    theMerger = new GsfVertexMerger(mergerPSet);
  }

  theFitter = new AdaptiveVertexFitter(
      GeometricAnnealing(),
      linP,
      GsfVertexUpdator(limitComponents_, &*theMerger),
      GsfVertexTrackCompatibilityEstimator(),
      GsfVertexSmoother(limitComponents_, &*theMerger),
      MultiPerigeeLTSFactory() );
 
//   theFitter->setMaximumDistance(theMaxShift);
//   theFitter->setMaximumNumberOfIterations(theMaxStep);
}

AdaptiveGsfVertexFitter::AdaptiveGsfVertexFitter(const AdaptiveGsfVertexFitter & original)
{
  theFitter = original.theFitter->clone();
}

AdaptiveGsfVertexFitter::~AdaptiveGsfVertexFitter()
{
  delete theFitter;
}
