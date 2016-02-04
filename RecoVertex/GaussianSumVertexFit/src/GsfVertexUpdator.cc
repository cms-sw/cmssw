#include "RecoVertex/GaussianSumVertexFit/interface/GsfVertexUpdator.h"
#include "RecoVertex/GaussianSumVertexFit/interface/BasicMultiVertexState.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <cfloat>

GsfVertexUpdator::GsfVertexUpdator(bool limit, const GsfVertexMerger * merger) :
  limitComponents (limit)
{
  if (limitComponents) theMerger = merger->clone();
}


CachingVertex<5> GsfVertexUpdator::add(const CachingVertex<5> & oldVertex,
				    const RefCountedVertexTrack track) const
{

  VSC prevVtxComponents = oldVertex.vertexState().components();

//   cout << "GsfVertexUpdator::Add new Track with "
//       << track->linearizedTrack()->components().size() << " components to vertex of "
//       << prevVtxComponents.size() << " components.\n";

  if (prevVtxComponents.empty()) {
  throw VertexException
    ("GsfVertexUpdator::(Previous) Vertex to update has no components");
  }

  LTC ltComponents = track->linearizedTrack()->components();
  if (ltComponents.empty()) {
  throw VertexException
    ("GsfVertexUpdator::Track to add to vertex has no components");
  }

  if ((ltComponents.size()==1) && (prevVtxComponents.size()==1)) 
    return kalmanVertexUpdator.add(oldVertex, track);

  float trackWeight = track->weight();

  std::vector<VertexComponent> newVertexComponents;
  newVertexComponents.reserve(prevVtxComponents.size()*ltComponents.size());

//     for (LTC::iterator trackCompIter = ltComponents.begin();
//   	trackCompIter != ltComponents.end(); trackCompIter++ ) {
//   cout <<(**trackCompIter).state().globalPosition()<<endl;
//     }

  for (VSC::iterator vertexCompIter = prevVtxComponents.begin();
  	vertexCompIter != prevVtxComponents.end(); vertexCompIter++ ) {
    for (LTC::iterator trackCompIter = ltComponents.begin();
  	trackCompIter != ltComponents.end(); trackCompIter++ ) {
      newVertexComponents.push_back(
        createNewComponent(*vertexCompIter, *trackCompIter, trackWeight, +1));
	 // return invalid vertex in case one of the updated vertex-components is invalid
      if (!newVertexComponents.back().first.isValid()) return CachingVertex<5>();
    }
  }
//   cout << "updator components: "<<newVertexComponents.size()<<endl;

  // Update tracks vector

  std::vector<RefCountedVertexTrack> newVertexTracks = oldVertex.tracks();
  newVertexTracks.push_back(track);
//   cout << "a \n ";

  // Assemble VertexStates and compute Chi**2

  VertexChi2Pair vertexChi2Pair = assembleVertexComponents(newVertexComponents);
//   cout << "b \n ";
  VertexState newVertexState = vertexChi2Pair.first;
//   cout << "c \n ";
  double chi2 = oldVertex.totalChiSquared() + vertexChi2Pair.second;

  // Merge:
  if (limitComponents) newVertexState = theMerger->merge(newVertexState);

  if  (oldVertex.hasPrior()) {
    return CachingVertex<5>(oldVertex.priorPosition(), oldVertex.priorError(),
    		newVertexState.weightTimesPosition(),
		newVertexState.weight(), newVertexTracks, chi2);
  } else {
    return CachingVertex<5>(newVertexState, newVertexTracks, chi2);
  }


}




CachingVertex<5> GsfVertexUpdator::remove(const CachingVertex<5> & oldVertex, 
	const RefCountedVertexTrack track) const
{
  throw VertexException("GsfVertexUpdator::Remove Methode not yet done");
//  return CachingVertex<5>();
}


  /**
   * Where one component of the previous vertex gets updated with one component of
   *  the track.
   */

GsfVertexUpdator::VertexComponent 
GsfVertexUpdator::createNewComponent(const VertexState & oldVertex,
	 const RefCountedLinearizedTrackState linTrack, float weight, int sign) const
{

  if(abs(sign) != 1)
    throw VertexException ("GsfVertexUpdator::sign not equal to 1.");

  if(sign == -1)
    throw VertexException("GsfVertexUpdator::sign of -1 not yet implemented.");


  // Weight of the component in the mixture (non-normalized)
  double weightInMixture = theWeightCalculator.calculate(oldVertex, linTrack, 1.E9);
  if (weightInMixture < 0.) return VertexComponent(VertexState(), WeightChi2Pair(0.,0.));

  // position estimate of the component
  VertexState newVertex = kalmanVertexUpdator.positionUpdate(oldVertex, 
  				linTrack, weight, sign);
  if (!newVertex.isValid()) return VertexComponent(newVertex, WeightChi2Pair(0.,0.));

  //Chi**2 contribution of the component
  std::pair <bool, double> chi2P = kalmanVertexUpdator.chi2Increment(oldVertex, newVertex, 
  				linTrack, weight);
  if (!chi2P.first) return VertexComponent(VertexState(), WeightChi2Pair(0.,0.));
//         cout << "Update: "<<oldVertex.position()<<" "<<newVertex.position()<<" "<<chi2
// 	     <<" "<<linTrack->weightInMixture()<<" "<<weightInMixture<<endl;

  return VertexComponent(newVertex, WeightChi2Pair(weightInMixture, chi2P.second));
}

GsfVertexUpdator::VertexChi2Pair GsfVertexUpdator::assembleVertexComponents(
	const std::vector<GsfVertexUpdator::VertexComponent> & newVertexComponents) const
{
  VSC vertexComponents;
  vertexComponents.reserve(newVertexComponents.size());
  
  //renormalize weights
// cout << "assemble "<<newVertexComponents.size()<<endl;
  double totalWeight = 0.;
  double totalChi2 = 0.;

  for (std::vector<VertexComponent>::const_iterator iter = newVertexComponents.begin();
    iter != newVertexComponents.end(); iter ++) {
    totalWeight += iter->second.first;
  }
// cout << "totalWeight "<<totalWeight<<endl;
  if (totalWeight<DBL_MIN) {
    edm::LogWarning("GsfVertexUpdator") << "Updated Vertex has total weight of 0. "
    <<"The track is probably very far away.";
    return VertexChi2Pair( VertexState(), 0.);
  }

  for (std::vector<VertexComponent>::const_iterator iter = newVertexComponents.begin();
    iter != newVertexComponents.end(); iter ++) {
    double weight = iter->second.first/totalWeight;
    if (iter->second.first>DBL_MIN) {
      vertexComponents.push_back(VertexState(iter->first.weightTimesPosition(),
       iter->first.weight(), weight));
      totalChi2 += iter->second.second * weight;
    }
  }
// cout << "totalChi2 "<<totalChi2<<endl;
// cout << "vertexComponents "<<vertexComponents.size()<<endl;

  if (vertexComponents.empty()){
    edm::LogWarning("GsfVertexUpdator") << "No Vertex State left after reweighting.";
    return VertexChi2Pair( VertexState(), 0.);
  }

  return VertexChi2Pair( VertexState( new BasicMultiVertexState( vertexComponents)),
  			totalChi2);
}
