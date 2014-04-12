#include "RecoVertex/GaussianSumVertexFit/interface/GsfVertexMerger.h"
#include "TrackingTools/GsfTools/interface/KullbackLeiblerDistance.h"
// #include "TrackingTools/GsfTools/interface/MahalanobisDistance.h"
#include "TrackingTools/GsfTools/interface/CloseComponentsMerger.h"
// #include "CommonReco/GSFTools/interface/KeepingNonZeroWeightsMerger.h"
// #include "TrackingTools/GsfTools/interface/LargestWeightsStateMerger.h"
// #include "TrackingTools/GsfTools/interface/RCMultiGaussianState.h"
#include "DataFormats/Common/interface/Handle.h"
#include "RecoVertex/VertexPrimitives/interface/VertexException.h"

#include "RecoVertex/GaussianSumVertexFit/interface/VertexGaussianStateConversions.h"

GsfVertexMerger::GsfVertexMerger(const edm::ParameterSet& pSet)
{

  maxComponents = pSet.getParameter<int>("maxNbrComponents");
  std::string mergerName = pSet.getParameter<std::string>("merger");
  std::string distanceName = pSet.getParameter<std::string>("distance");

  if ( mergerName=="CloseComponentsMerger" ) {
    DistanceBetweenComponents<3>* distance;

    if ( distanceName=="KullbackLeiblerDistance" )
      distance = new KullbackLeiblerDistance<3>();
//     else if ( distanceName=="MahalanobisDistance" )
//       distance = new MahalanobisDistance();
    else 
      throw VertexException("GsfVertexMerger: Distance type "+distanceName+" unknown. Check distance parameter in GsfMergerParameters PSet");
    
    merger = new CloseComponentsMerger<3>(maxComponents, distance);
    delete distance;
  } 
//   else if ( mergerName=="LargestWeightsStateMerger" )
//     merger = new LargestWeightsStateMerger(maxComponents);
  else 
    throw VertexException("GsfVertexMerger: Merger type "+mergerName+" unknown. Check merger parameter in GsfMergerParameters PSet");

//   else if ( mergerName=="KeepingNonZeroWeightsMerger" )
//     merger = new KeepingNonZeroWeightsMerger();

//   std::string mergerName = "test";
//   edm::ESHandle<MultiGaussianStateMerger> mergerProducer;
//   iRecord.get(mergerName,mergerProducer);
//   merger = (MultiGaussianStateMerger *) mergerProducer.product();

}

CachingVertex<5> GsfVertexMerger::merge(const CachingVertex<5> & oldVertex) const
{
  if (oldVertex.vertexState().components().size() <= maxComponents) 
  	return oldVertex;

  VertexState newVertexState = merge(oldVertex.vertexState());

  if  (oldVertex.hasPrior()) {
    return CachingVertex<5>(oldVertex.priorPosition(), oldVertex.priorError(),
    		newVertexState.weightTimesPosition(), newVertexState.weight(),
		oldVertex.tracks(), oldVertex.totalChiSquared());
  } else {
    return CachingVertex<5>(newVertexState, oldVertex.tracks(), 
    		oldVertex.totalChiSquared());
  }
}


VertexState GsfVertexMerger::merge(const VertexState & oldVertex) const
{
  using namespace GaussianStateConversions;

  if (oldVertex.components().size() <= maxComponents) 
  	return oldVertex;

  MultiGaussianState<3> multiGaussianState(multiGaussianStateFromVertex(oldVertex));
  MultiGaussianState<3> finalState(merger->merge(multiGaussianState));
  return vertexFromMultiGaussianState(finalState);
}
