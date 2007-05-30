#include "RecoVertex/VertexTools/interface/SequentialVertexFitter.h"
#include "DataFormats/GeometryCommonDetAlgo/interface/GlobalError.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <algorithm>
using namespace std;

namespace {
  // FIXME
  // hard-coded tracker bounds
  // workaround while waiting for Geometry service
  static const float TrackerBoundsRadius = 112;
  static const float TrackerBoundsHalfLength = 273.5;
  bool insideTrackerBounds(const GlobalPoint& point) {
    return ((point.transverse() < TrackerBoundsRadius)
        && (abs(point.z()) < TrackerBoundsHalfLength));
  }
}


SequentialVertexFitter::SequentialVertexFitter(
  const LinearizationPointFinder & linP, 
  const VertexUpdator & updator, const VertexSmoother & smoother,
  const AbstractLTSFactory & ltsf ) : 
  theLinP(linP.clone()), theUpdator(updator.clone()), 
  theSmoother(smoother.clone()), theLTrackFactory ( ltsf.clone() )
{
  setDefaultParameters();
}

SequentialVertexFitter::SequentialVertexFitter(
  const edm::ParameterSet& pSet, const LinearizationPointFinder & linP, 
  const VertexUpdator & updator, const VertexSmoother & smoother,
  const AbstractLTSFactory & ltsf) : 
  thePSet(pSet), theLinP(linP.clone()), theUpdator(updator.clone()), 
  theSmoother(smoother.clone()), theLTrackFactory ( ltsf.clone() )
{
  readParameters();
}


SequentialVertexFitter::SequentialVertexFitter(
  const SequentialVertexFitter & original)
{
  thePSet = original.parameterSet();
  theLinP = original.linearizationPointFinder()->clone();
  theUpdator = original.vertexUpdator()->clone();
  theSmoother = original.vertexSmoother()->clone();
  theMaxShift = original.maxShift();
  theMaxStep = original.maxStep();
  theLTrackFactory = original.linearizedTrackStateFactory()->clone();
}


SequentialVertexFitter::~SequentialVertexFitter()
{
  delete theLinP;
  delete theUpdator;
  delete theSmoother;
  delete theLTrackFactory;
}


void SequentialVertexFitter::readParameters()
{
  theMaxShift = thePSet.getParameter<double>("maxDistance"); //0.01
  theMaxStep = thePSet.getParameter<int>("maxNbrOfIterations"); //10
}

void SequentialVertexFitter::setDefaultParameters()
{
  thePSet.addParameter<double>("maxDistance", 0.01);
  thePSet.addParameter<int>("maxNbrOfIterations", 10); //10
  readParameters();
}

CachingVertex 
SequentialVertexFitter::vertex(const vector<reco::TransientTrack> & tracks) const
{
  // Linearization Point
  GlobalPoint linP = theLinP->getLinearizationPoint(tracks);
  if (!insideTrackerBounds(linP)) linP = GlobalPoint(0,0,0);

  // Initial vertex state, with a very large error matrix
  AlgebraicSymMatrix we(3,1);
  GlobalError error(we*10000);
  VertexState state(linP, error);
  vector<RefCountedVertexTrack> vtContainer = linearizeTracks(tracks, state);
  return fit(vtContainer, state, false);
}


CachingVertex 
SequentialVertexFitter::vertex(const vector<RefCountedVertexTrack> & tracks) const
{
  // Initial vertex state, with a very small weight matrix
  GlobalPoint linP = tracks[0]->linearizedTrack()->linearizationPoint();
  AlgebraicSymMatrix we(3,1);
  GlobalError error(we*10000);
  VertexState state(linP, error);
  return fit(tracks, state, false);
}


// Fit vertex out of a set of RecTracks. 
// Uses the specified linearization point.
//
CachingVertex  
SequentialVertexFitter::vertex(const vector<reco::TransientTrack> & tracks, 
			       const GlobalPoint& linPoint) const
{ 
  // Initial vertex state, with a very large error matrix
  AlgebraicSymMatrix we(3,1);
  GlobalError error(we*10000);
  VertexState state(linPoint, error);
  vector<RefCountedVertexTrack> vtContainer = linearizeTracks(tracks, state);
  return fit(vtContainer, state, false);
}


  /** Fit vertex out of a set of TransientTracks. 
   *  The specified BeamSpot will be used as priot, but NOT for the linearization.
   * The specified LinearizationPointFinder will be used to find the linearization point.
   */
CachingVertex 
SequentialVertexFitter::vertex(const vector<reco::TransientTrack> & tracks,
			       const BeamSpot& beamSpot) const
{
  VertexState beamSpotState(beamSpot.position(), beamSpot.error());
  vector<RefCountedVertexTrack> vtContainer;

  if (tracks.size() > 1) {
    // Linearization Point search if there are more than 1 track
    GlobalPoint linP = theLinP->getLinearizationPoint(tracks);
    if (!insideTrackerBounds(linP)) linP = GlobalPoint(0,0,0);
    AlgebraicSymMatrix we(3,1);
    GlobalError error(we*10000);
    VertexState lpState(linP, error);
    vtContainer = linearizeTracks(tracks, lpState);
  } else {
    // otherwise take the beamspot position.
    vtContainer = linearizeTracks(tracks, beamSpotState);
  }

  return fit(vtContainer, beamSpotState, true);
}


// Fit vertex out of a set of RecTracks. 
// Uses the position as both the linearization point AND as prior
// estimate of the vertex position. The error is used for the 
// weight of the prior estimate.
//
CachingVertex SequentialVertexFitter::vertex(
  const vector<reco::TransientTrack> & tracks, 
  const GlobalPoint& priorPos,
  const GlobalError& priorError) const
{ 
  VertexState state(priorPos, priorError);
  vector<RefCountedVertexTrack> vtContainer = linearizeTracks(tracks, state);
  return fit(vtContainer, state, true);
}

// Fit vertex out of a set of VertexTracks
// Uses the position and error for the prior estimate of the vertex.
// This position is not used to relinearize the tracks.
//
CachingVertex SequentialVertexFitter::vertex(
  const vector<RefCountedVertexTrack> & tracks, 
  const GlobalPoint& priorPos,
  const GlobalError& priorError) const
{
  VertexState state(priorPos, priorError);
  return fit(tracks, state, true);
}


// Construct a container of VertexTrack from a set of RecTracks.
//
vector<RefCountedVertexTrack> 
SequentialVertexFitter::linearizeTracks(
  const vector<reco::TransientTrack> & tracks, 
  const VertexState state) const
{
  GlobalPoint linP = state.position();
  vector<RefCountedVertexTrack> finalTracks;
  finalTracks.reserve(tracks.size());
  for(vector<reco::TransientTrack>::const_iterator i = tracks.begin(); 
      i != tracks.end(); i++) {
    RefCountedLinearizedTrackState lTrData 
      = theLTrackFactory->linearizedTrackState(linP, *i);
    RefCountedVertexTrack vTrData = theVTrackFactory.vertexTrack(lTrData,state);
    finalTracks.push_back(vTrData);
  }
  return finalTracks;
}


// Construct new a container of VertexTrack with a new linearization point
// and vertex state, from an existing set of VertexTrack, from which only the 
// recTracks will be used.
//
vector<RefCountedVertexTrack> 
SequentialVertexFitter::reLinearizeTracks(
  const vector<RefCountedVertexTrack> & tracks, 
  const VertexState state) const
{

  GlobalPoint linP = state.position();
  vector<RefCountedVertexTrack> finalTracks;
  finalTracks.reserve(tracks.size());
  for(vector<RefCountedVertexTrack>::const_iterator i = tracks.begin(); 
      i != tracks.end(); i++) {
    RefCountedLinearizedTrackState lTrData = 
    	  (**i).linearizedTrack()->stateWithNewLinearizationPoint(linP);
    //    RefCountedLinearizedTrackState lTrData = 
    //      theLTrackFactory->linearizedTrackState(linP, 
    // 				    (**i).linearizedTrack()->track());
    RefCountedVertexTrack vTrData =
      theVTrackFactory.vertexTrack(lTrData,state, (**i).weight() );
    finalTracks.push_back(vTrData);
  }
  return finalTracks;
}


// The method where the vertex fit is actually done!
//
CachingVertex 
SequentialVertexFitter::fit(const vector<RefCountedVertexTrack> & tracks,
  			    const VertexState priorVertex,
			    bool withPrior) const
{
  vector<RefCountedVertexTrack> initialTracks;
  GlobalPoint priorVertexPosition = priorVertex.position();
  GlobalError priorVertexError = priorVertex.error();
  
  CachingVertex returnVertex(priorVertexPosition,priorVertexError,initialTracks,0);
  if (withPrior) {
    returnVertex = CachingVertex(priorVertexPosition,priorVertexError,
    		priorVertexPosition,priorVertexError,initialTracks,0);
  }
  CachingVertex initialVertex = returnVertex;
  vector<RefCountedVertexTrack> globalVTracks = tracks;

  // main loop through all the VTracks
  bool validVertex = true;
  int step = 0;
  GlobalPoint newPosition = priorVertexPosition;
  GlobalPoint previousPosition;
  do {
    CachingVertex fVertex = initialVertex;
    // make new linearized and vertex tracks for the next iteration
    if(step != 0) globalVTracks = reLinearizeTracks(tracks, 
    					returnVertex.vertexState());

    // update sequentially the vertex estimate
    for (vector<RefCountedVertexTrack>::const_iterator i 
	   = globalVTracks.begin(); i != globalVTracks.end(); i++) {
      fVertex = theUpdator->add(fVertex,*i);
    }

    validVertex = true;
    // check tracker bounds and NaN in position
    if (!insideTrackerBounds(fVertex.position())) {
      edm::LogError("RecoVertex/SequentialVertexFitter") 
	 << "Fitted position is out of tracker bounds.\n";
      validVertex = false;
    }
    
    if (hasNan(fVertex.position())) {
      edm::LogError("RecoVertex/SequentialVertexFitter") 
	 << "Fitted position is NaN.\n";
      validVertex = false;
    }

    if (!validVertex) {
      // reset initial vertex position to (0,0,0) and force new iteration 
      // if number of steps not exceeded
      fVertex = CachingVertex(GlobalPoint(0,0,0), fVertex.error(),
                              initialTracks, 0);
    }

    previousPosition = newPosition;
    newPosition = fVertex.position();

    returnVertex = fVertex;
    globalVTracks.clear();
    step++;
  } while ( (step != theMaxStep) &&
  	    (((previousPosition - newPosition).transverse() > theMaxShift) ||
		(!validVertex) ) );

  if (!validVertex) {
    edm::LogError("RecoVertex/SequentialVertexFitter") 
       << "Fitted position is invalid (out of tracker bounds or has NaN). Returned vertex is invalid\n";
    return CachingVertex(); // return invalid vertex
  }

  if (step >= theMaxStep) {
    edm::LogError("RecoVertex/SequentialVertexFitter") 
       << "The maximum number of steps has been exceeded. Returned vertex is invalid\n";
    return CachingVertex(); // return invalid vertex
  }

  // smoothing
  returnVertex = theSmoother->smooth(returnVertex);

  return returnVertex;
}
