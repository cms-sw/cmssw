#include "RecoVertex/GaussianSumVertexFit/interface/GsfVertexMerger.h"
#include "TrackingTools/GsfTools/interface/KullbackLeiblerDistance.h"
#include "TrackingTools/GsfTools/interface/MahalanobisDistance.h"
#include "TrackingTools/GsfTools/interface/CloseComponentsMerger.h"
// #include "CommonReco/GSFTools/interface/KeepingNonZeroWeightsMerger.h"
#include "TrackingTools/GsfTools/interface/LargestWeightsStateMerger.h"
#include "RecoVertex/GaussianSumVertexFit/interface/RCGaussianStateFactory.h"
#include "TrackingTools/GsfTools/interface/RCMultiGaussianState.h"
#include "RecoVertex/VertexPrimitives/interface/CachingVertex.h"
#include "DataFormats/Common/interface/EDProduct.h"
#include "FWCore/Framework/interface/Handle.h"
#include "RecoVertex/VertexPrimitives/interface/VertexException.h"

GsfVertexMerger::GsfVertexMerger(const edm::ParameterSet& pSet)
{

  maxComponents = pSet.getParameter<int>("maxNbrComponents");
  std::string mergerName = pSet.getParameter<string>("merger");
  std::string distanceName = pSet.getParameter<string>("distance");

  if ( mergerName=="CloseComponentsMerger" ) {
    DistanceBetweenComponents* distance;

    if ( distanceName=="KullbackLeiblerDistance" )
      distance = new KullbackLeiblerDistance();
    else if ( distanceName=="MahalanobisDistance" )
      distance = new MahalanobisDistance();
    else 
      throw VertexException("GsfVertexMerger: Distance type "+distanceName+" unknown. Check distance parameter in GsfMergerParameters PSet");
    
    merger = new CloseComponentsMerger(maxComponents, distance);
    delete distance;
  } else if ( mergerName=="LargestWeightsStateMerger" )
    merger = new LargestWeightsStateMerger(maxComponents);
  else 
    throw VertexException("GsfVertexMerger: Merger type "+mergerName+" unknown. Check merger parameter in GsfMergerParameters PSet");

//   else if ( mergerName=="KeepingNonZeroWeightsMerger" )
//     merger = new KeepingNonZeroWeightsMerger();

//   std::string mergerName = "test";
//   edm::ESHandle<MultiGaussianStateMerger> mergerProducer;
//   iRecord.get(mergerName,mergerProducer);
//   merger = (MultiGaussianStateMerger *) mergerProducer.product();

}

CachingVertex GsfVertexMerger::merge(const CachingVertex & oldVertex) const
{
  if (oldVertex.vertexState().components().size() <= maxComponents) 
  	return oldVertex;
cout << "start merger :"<<oldVertex.vertexState().components().size()<<endl;
  VertexState newVertexState = merge(oldVertex.vertexState());
cout << "end merger :"<<newVertexState.components().size()<<endl;

  if  (oldVertex.hasPrior()) {
    return CachingVertex(oldVertex.priorPosition(), oldVertex.priorError(),
    		newVertexState.weightTimesPosition(), newVertexState.weight(),
		oldVertex.tracks(), oldVertex.totalChiSquared());
  } else {
    return CachingVertex(newVertexState, oldVertex.tracks(), 
    		oldVertex.totalChiSquared());
  }
}


VertexState GsfVertexMerger::merge(const VertexState & oldVertex) const
{
  if (oldVertex.components().size() <= maxComponents) 
  	return oldVertex;

  RCGaussianStateFactory theFactory;
  RCMultiGaussianState multiGaussianState = 
  	theFactory.multiGaussianState(oldVertex);
  RCMultiGaussianState finalState = merger->merge(multiGaussianState);

  const MultiGaussianState * finalStatePtr = finalState.get();
  const MultiGaussianStateFromVertex* temp = 
 	dynamic_cast<const MultiGaussianStateFromVertex*>(finalStatePtr);

  if (temp == 0) {
   cout << "PerigeeLinearizedTrackState: finalState is not a MultiGaussianStateFromVertex\n";
   throw VertexException("PerigeeLinearizedTrackState: finalState is not a MultiGaussianStateFromVertex");
  }

  VertexState newVertexState = temp->state();
  return newVertexState;
}
