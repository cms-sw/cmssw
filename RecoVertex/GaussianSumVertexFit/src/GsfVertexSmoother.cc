#include "RecoVertex/GaussianSumVertexFit/interface/GsfVertexSmoother.h"
#include "RecoVertex/GaussianSumVertexFit/interface/BasicMultiVertexState.h"
#include "RecoVertex/GaussianSumVertexFit/interface/MultiRefittedTS.h"
#include "RecoVertex/VertexPrimitives/interface/VertexException.h"

GsfVertexSmoother::GsfVertexSmoother(bool limit, const GsfVertexMerger * merger) :
  limitComponents (limit)
{
  if (limitComponents) theMerger = merger->clone();
}

CachingVertex<5>
GsfVertexSmoother::smooth(const CachingVertex<5> & vertex) const
{

  std::vector<RefCountedVertexTrack> tracks = vertex.tracks();
  int numberOfTracks = tracks.size();
  if (numberOfTracks<1) return vertex;

  // Initial vertex for ascending fit
  GlobalPoint priorVertexPosition = tracks[0]->linearizedTrack()->linearizationPoint();
  AlgebraicSymMatrix we(3,1);
  GlobalError priorVertexError(we*10000);

  std::vector<RefCountedVertexTrack> initialTracks;
  CachingVertex<5> fitVertex(priorVertexPosition,priorVertexError,initialTracks,0);
  //In case prior vertex was used.
  if (vertex.hasPrior()) {
    VertexState priorVertexState = vertex.priorVertexState();
    fitVertex = CachingVertex<5>(priorVertexState, priorVertexState,
    		initialTracks,0);
  }

  // vertices from ascending fit
  std::vector<CachingVertex<5> > ascendingFittedVertices;
  ascendingFittedVertices.reserve(numberOfTracks);
  ascendingFittedVertices.push_back(fitVertex);

  // ascending fit
  for (std::vector<RefCountedVertexTrack>::const_iterator i = tracks.begin();
	  i != (tracks.end()-1); ++i) {
    fitVertex = theUpdator.add(fitVertex,*i);
    if (limitComponents) fitVertex = theMerger->merge(fitVertex);
    ascendingFittedVertices.push_back(fitVertex);
  }

  // Initial vertex for descending fit
  priorVertexPosition = tracks[0]->linearizedTrack()->linearizationPoint();
  priorVertexError = GlobalError(we*10000);
  fitVertex = CachingVertex<5>(priorVertexPosition,priorVertexError,initialTracks,0);

  // vertices from descending fit
  std::vector<CachingVertex<5> > descendingFittedVertices;
  descendingFittedVertices.reserve(numberOfTracks);
  descendingFittedVertices.push_back(fitVertex);

  // descending fit
  for (std::vector<RefCountedVertexTrack>::const_iterator i = (tracks.end()-1);
	  i != (tracks.begin()); --i) {
    fitVertex = theUpdator.add(fitVertex,*i);
    if (limitComponents) fitVertex = theMerger->merge(fitVertex);
    descendingFittedVertices.insert(descendingFittedVertices.begin(), fitVertex);
  }

  std::vector<RefCountedVertexTrack> newTracks;
  double smoothedChi2 = 0.;  // Smoothed chi2

  // Track refit
  for(std::vector<RefCountedVertexTrack>::const_iterator i = tracks.begin();
  	i != tracks.end();i++)
  {
    int indexNumber = i-tracks.begin();
    //First, combine the vertices:
    VertexState meanedVertex = 
         meanVertex(ascendingFittedVertices[indexNumber].vertexState(), 
    		    descendingFittedVertices[indexNumber].vertexState());
    if (limitComponents) meanedVertex = theMerger->merge(meanedVertex);
    // Add the vertex and smooth the track:
    TrackChi2Pair thePair = vertexAndTrackUpdate(meanedVertex, *i, vertex.position());
    smoothedChi2 += thePair.second.second;
    newTracks.push_back( theVTFactory.vertexTrack((**i).linearizedTrack(),
  	vertex.vertexState(), thePair.first, thePair.second.second,
	AlgebraicSymMatrixOO(), (**i).weight()) );
  }

  if  (vertex.hasPrior()) {
    smoothedChi2 += priorVertexChi2(vertex.priorVertexState(), vertex.vertexState());
    return CachingVertex<5>(vertex.priorVertexState(), vertex.vertexState(),
    		newTracks, smoothedChi2);
  } else {
    return CachingVertex<5>(vertex.vertexState(), newTracks, smoothedChi2);
  }
}

GsfVertexSmoother::TrackChi2Pair 
GsfVertexSmoother::vertexAndTrackUpdate(const VertexState & oldVertex,
	const RefCountedVertexTrack track, const GlobalPoint & referencePosition) const
{

  VSC prevVtxComponents = oldVertex.components();

  if (prevVtxComponents.empty()) {
  throw VertexException
    ("GsfVertexSmoother::(Previous) Vertex to update has no components");
  }

  LTC ltComponents = track->linearizedTrack()->components();
  if (ltComponents.empty()) {
  throw VertexException
    ("GsfVertexSmoother::Track to add to vertex has no components");
  }
  float trackWeight = track->weight();

  std::vector<RefittedTrackComponent> newTrackComponents;
  newTrackComponents.reserve(prevVtxComponents.size()*ltComponents.size());

  for (VSC::iterator vertexCompIter = prevVtxComponents.begin();
  	vertexCompIter != prevVtxComponents.end(); vertexCompIter++ ) {

    for (LTC::iterator trackCompIter = ltComponents.begin();
  	trackCompIter != ltComponents.end(); trackCompIter++ ) {
      newTrackComponents.push_back
        (createNewComponent(*vertexCompIter, *trackCompIter, trackWeight));
    }
  }

  return assembleTrackComponents(newTrackComponents, referencePosition);
}

/**
 * This method assembles all the components of the refitted track into one refitted track state,
 * normalizing the components. Also, it adds the chi2 track-components increments.
 */

GsfVertexSmoother::TrackChi2Pair GsfVertexSmoother::assembleTrackComponents(
	const std::vector<GsfVertexSmoother::RefittedTrackComponent> & trackComponents,
	const GlobalPoint & referencePosition)
	const
{

  //renormalize weights

  double totalWeight = 0.;
  double totalVtxChi2 = 0., totalTrkChi2 = 0.;

  for (std::vector<RefittedTrackComponent>::const_iterator iter = trackComponents.begin();
    iter != trackComponents.end(); ++iter ) {
    totalWeight += iter->first.second;
    totalVtxChi2 += iter->second.first  * iter->first.second ;
    totalTrkChi2 += iter->second.second * iter->first.second ;
  }

  totalVtxChi2 /= totalWeight ;
  totalTrkChi2 /= totalWeight ;

  std::vector<RefCountedRefittedTrackState> reWeightedRTSC;
  reWeightedRTSC.reserve(trackComponents.size());
  

  for (std::vector<RefittedTrackComponent>::const_iterator iter = trackComponents.begin();
    iter != trackComponents.end(); ++iter ) {
    if (iter->second.first!=0) {
      reWeightedRTSC.push_back(iter->first.first->stateWithNewWeight(iter->second.first/totalWeight));
    }
  }

  RefCountedRefittedTrackState finalRTS = 
    RefCountedRefittedTrackState(new MultiRefittedTS(reWeightedRTSC, referencePosition));
  return TrackChi2Pair(finalRTS, VtxTrkChi2Pair(totalVtxChi2, totalTrkChi2));
}


  /**
   * This method does the smoothing of one track component with one vertex component.
   * And the track-component-chi2 increment and weight of new component in mixture.
   */

GsfVertexSmoother::RefittedTrackComponent 
GsfVertexSmoother::createNewComponent(const VertexState & oldVertex,
	 const RefCountedLinearizedTrackState linTrack, float trackWeight) const
{

  int sign =+1;

  // Weight of the component in the mixture (non-normalized)
  double weightInMixture = theWeightCalculator.calculate(oldVertex, linTrack, 1000000000.);

  // position estimate of the component
  VertexState newVertex = kalmanVertexUpdator.positionUpdate(oldVertex,
	linTrack, trackWeight, sign);

  KalmanVertexTrackUpdator<5>::trackMatrixPair thePair = 
  	theVertexTrackUpdator.trackRefit(newVertex, linTrack, trackWeight);

  //Chi**2 contribution of the track component
  double vtxChi2 = helper.vertexChi2(oldVertex, newVertex);
  std::pair<bool, double> trkCi2 = helper.trackParameterChi2(linTrack, thePair.first);

  return RefittedTrackComponent(TrackWeightPair(thePair.first, weightInMixture), 
  			VtxTrkChi2Pair(vtxChi2, trkCi2.second));
}


VertexState
GsfVertexSmoother::meanVertex(const VertexState & vertexA,
			      const VertexState & vertexB) const
{
  std::vector<VertexState> vsCompA = vertexA.components();
  std::vector<VertexState> vsCompB = vertexB.components();
  std::vector<VertexState> finalVS;
  finalVS.reserve(vsCompA.size()*vsCompB.size());
  for (std::vector<VertexState>::iterator iA = vsCompA.begin(); iA!= vsCompA.end(); ++iA)
  {
    for (std::vector<VertexState>::iterator iB = vsCompB.begin(); iB!= vsCompB.end(); ++iB)
    {
      AlgebraicSymMatrix33 newWeight = iA->weight().matrix_new() +
				     iB->weight().matrix_new();
      AlgebraicVector3 newWtP = iA->weightTimesPosition() +
      			       iB->weightTimesPosition();
      double newWeightInMixture = iA->weightInMixture() *
				  iB->weightInMixture();
      finalVS.push_back( VertexState(newWtP, newWeight, newWeightInMixture) );
    }
  }
  #ifndef CMS_NO_COMPLEX_RETURNS
    return VertexState(new BasicMultiVertexState(finalVS));
  #else
    VertexState theFinalVM(new BasicMultiVertexState(finalVS));
    return theFinalVM;
  #endif
}


double GsfVertexSmoother::priorVertexChi2(
	const VertexState priorVertex, const VertexState fittedVertex) const
{
  std::vector<VertexState> priorVertexComp  = priorVertex.components();
  std::vector<VertexState> fittedVertexComp = fittedVertex.components();
  double vetexChi2 = 0.;
  for (std::vector<VertexState>::iterator pvI = priorVertexComp.begin(); 
  	pvI!= priorVertexComp.end(); ++pvI)
  {
    for (std::vector<VertexState>::iterator fvI = fittedVertexComp.begin(); 
    	fvI!= fittedVertexComp.end(); ++fvI)
    {
      vetexChi2 += (pvI->weightInMixture())*(fvI->weightInMixture())*
      			helper.vertexChi2(*pvI, *fvI);
    }
  }
  return vetexChi2;
}
