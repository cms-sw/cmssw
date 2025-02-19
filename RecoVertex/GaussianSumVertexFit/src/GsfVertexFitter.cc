#include "RecoVertex/GaussianSumVertexFit/interface/GsfVertexFitter.h"
#include "RecoVertex/VertexTools/interface/DummyVertexSmoother.h"
#include "RecoVertex/GaussianSumVertexFit/interface/GsfVertexSmoother.h"
#include "RecoVertex/GaussianSumVertexFit/interface/MultiPerigeeLTSFactory.h"



GsfVertexFitter::GsfVertexFitter(const edm::ParameterSet& pSet,
	const LinearizationPointFinder & linP )
{
  float theMaxShift = pSet.getParameter<double>("maxDistance"); //0.01
  int theMaxStep = pSet.getParameter<int>("maxNbrOfIterations"); //10
  bool limitComponents_ = pSet.getParameter<bool>("limitComponents");
  bool useSmoothing = pSet.getParameter<bool>("smoothTracks");

  VertexSmoother<5> * theSmoother;
  DeepCopyPointerByClone<GsfVertexMerger> theMerger;

  if (limitComponents_) {
    edm::ParameterSet mergerPSet = pSet.getParameter<edm::ParameterSet>("GsfMergerParameters");
    theMerger = new GsfVertexMerger(mergerPSet);
  }

  if (useSmoothing) theSmoother = new GsfVertexSmoother(limitComponents_, &*theMerger);
    else theSmoother = new DummyVertexSmoother<5>();

  theSequentialFitter = new SequentialVertexFitter<5>(linP, 
	GsfVertexUpdator(limitComponents_, &*theMerger),
	*theSmoother, MultiPerigeeLTSFactory());
  theSequentialFitter->setMaximumDistance(theMaxShift);
  theSequentialFitter->setMaximumNumberOfIterations(theMaxStep);

  delete theSmoother;
 }

GsfVertexFitter::GsfVertexFitter(const GsfVertexFitter & original)
{
  theSequentialFitter = original.theSequentialFitter->clone();
}

GsfVertexFitter::~GsfVertexFitter()
{
  delete theSequentialFitter;
}
