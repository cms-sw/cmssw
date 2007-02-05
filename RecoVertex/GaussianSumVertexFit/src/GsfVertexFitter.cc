#include "RecoVertex/GaussianSumVertexFit/interface/GsfVertexFitter.h"
#include "Geometry/CommonDetAlgo/interface/GlobalError.h"
#include "RecoVertex/VertexTools/interface/DummyVertexSmoother.h"
#include "RecoVertex/GaussianSumVertexFit/interface/GsfVertexSmoother.h"

#include <algorithm>

GsfVertexFitter::GsfVertexFitter(const edm::ParameterSet& pSet,
	const LinearizationPointFinder & linP, bool useSmoothing) : theLinP(linP.clone())
{
cout <<"start\n";
  theMaxShift = pSet.getParameter<double>("maxDistance"); //0.01
  theMaxStep = pSet.getParameter<int>("maxNbrOfIterations"); //10
  limitComponents_ = pSet.getParameter<bool>("limitComponents");

  if (limitComponents_) {
    edm::ParameterSet mergerPSet = pSet.getParameter<edm::ParameterSet>("GsfMergerParameters");
    theMerger = new GsfVertexMerger(mergerPSet);
  }

  if (useSmoothing) theSmoother = new GsfVertexSmoother(limitComponents_, &*theMerger);
    else theSmoother = new DummyVertexSmoother();
cout <<"end\n";
}



GsfVertexFitter::GsfVertexFitter(const GsfVertexFitter & original)
{
  theLinP = original.linearizationPointFinder()->clone();
  theSmoother = original.vertexSmoother()->clone();
  theMaxShift = original.maxShift();
  theMaxStep = original.maxStep();
  limitComponents_ = original.limitComponents();
}


GsfVertexFitter::~GsfVertexFitter()
{
  delete theLinP;
  delete theSmoother;
}


// void GsfVertexFitter::readParameters(const edm::ParameterSet& pSet)
// {
// 
// //fixme : take PSet
// 
// //   static SimpleConfigurable<float>
// //     maxShiftSimTrackConfigurable(0.1,"GsfVertexFitter:maximumDistance");
// //   theMaxShift = maxShiftSimTrackConfigurable.value();
// // 
// //   static SimpleConfigurable<int>
// //     maxStepConfigurable(10,"GsfVertexFitter:maximumNumberOfIterations");
// //   theMaxStep = maxStepConfigurable.value();
// // 
// //   static SimpleConfigurable<bool> limitConfigurable(true, 
// //   		"GsfVertexFitter:limitComponents_");
// //   limitComponents_ = limitConfigurable.value();
// 
// }


CachingVertex 
GsfVertexFitter::vertex(const vector<reco::TransientTrack> & tracks) const
{ 
  // Linearization Point
  GlobalPoint linP = theLinP->getLinearizationPoint(tracks);

  // Initial vertex seed, with a very large error matrix
  AlgebraicSymMatrix we(3,1);
  GlobalError error(we*10000);
  VertexState state(linP, error);
  vector<RefCountedVertexTrack> vtContainer = linearizeTracks(tracks, state);
  return fit(vtContainer, state, false);
}


CachingVertex 
GsfVertexFitter::vertex(const vector<RefCountedVertexTrack> & tracks) const
{
  // Initial vertex seed, with a very small weight matrix
  GlobalPoint linP = tracks[0]->linearizedTrack()->linearizationPoint();
  AlgebraicSymMatrix we(3,1);
  GlobalError error(we*10000);
  VertexState state(linP, error);
  return fit(tracks, state, false);
}




// Fit vertex out of a set of reco::TransientTracks. 
// Uses the specified linearization point.
//
CachingVertex  
GsfVertexFitter::vertex(const vector<reco::TransientTrack> & tracks,
			const GlobalPoint& linPoint) const
{ 
  // Initial vertex seed, with a very large error matrix
  AlgebraicSymMatrix we(3,1);
  GlobalError error(we*10000);
  VertexState state(linPoint, error);
  vector<RefCountedVertexTrack> vtContainer = linearizeTracks(tracks, state);
  return fit(vtContainer, state, false);
}

// Fit vertex out of a set of reco::TransientTracks. 
// Uses the position as both the linearization point AND as prior
// estimate of the vertex position. The error is used for the 
// weight of the prior estimate.
//
CachingVertex GsfVertexFitter::vertex(
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
CachingVertex GsfVertexFitter::vertex(
  const vector<RefCountedVertexTrack> & tracks, 
  const GlobalPoint& priorPos,
  const GlobalError& priorError) const
{
  VertexState state(priorPos, priorError);
  return fit(tracks, state, true);
}

  /**
   * Construct a container of VertexTrack from a set of reco::TransientTracks.
   */

vector<RefCountedVertexTrack> 
GsfVertexFitter::linearizeTracks(const vector<reco::TransientTrack> & tracks, 
				 const VertexState state) const
{
  GlobalPoint linP = state.position();
  vector<RefCountedVertexTrack> finalTracks;
  finalTracks.reserve(tracks.size());
  for(vector<reco::TransientTrack>::const_iterator i = tracks.begin(); i != tracks.end(); i++)
  {
    RefCountedLinearizedTrackState lTrData 
      = lTrackFactory.linearizedTrackState(linP, *i);
    RefCountedVertexTrack vTrData = vTrackFactory.vertexTrack(lTrData,state);
   finalTracks.push_back(vTrData);
  }
  return finalTracks;
}


  /**
   * Construct new a container of VertexTrack with a new linearization point
   * and vertex seed, from an existing set of VertexTrack, from which only the 
   * reco::TransientTracks will be used.
   */

vector<RefCountedVertexTrack> 
GsfVertexFitter::reLinearizeTracks(const vector<RefCountedVertexTrack> & tracks, 
				   const VertexState state) const

{
  GlobalPoint linP = state.position();
  vector<RefCountedVertexTrack> finalTracks;
  finalTracks.reserve(tracks.size());
  for(vector<RefCountedVertexTrack>::const_iterator i = tracks.begin(); 
    i != tracks.end(); i++)
  {
//Fixme: check how much time is wasted by this...
//     RefCountedLinearizedTrackState lTrData = 
//     	(**i).linearizedTrack()->stateWithNewLinearizationPoint(linP);
    RefCountedLinearizedTrackState lTrData = 
      lTrackFactory.linearizedTrackState(linP, (**i).linearizedTrack()->track());
    RefCountedVertexTrack vTrData = vTrackFactory.vertexTrack(lTrData,state);
    finalTracks.push_back(vTrData);
  }
  return finalTracks;
}


// The method where the vertex fit is actually done!
//
CachingVertex 
GsfVertexFitter::fit(const vector<RefCountedVertexTrack> & tracks,
		     const VertexState priorVertex, bool withPrior) const
{
 cout << "Gsf fit method\n";
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
cout << "adding track "<<i-globalVTracks.begin()<<" left "<<globalVTracks.end()-i<<endl;
      fVertex = theUpdator.add(fVertex,*i);
      if (limitComponents_) fVertex = theMerger->merge(fVertex);
    }
    cout << "Step "<<step<< "position" << newPosition<<endl;
    previousPosition = newPosition;
    newPosition = fVertex.position();

    returnVertex = fVertex;
    globalVTracks.clear();
    step++;
  } while ( (step != theMaxStep) &&
  	    ((previousPosition - newPosition).transverse() > theMaxShift) );

  // smoothing
  returnVertex = theSmoother->smooth(returnVertex);
 cout <<"Bye!\n";
  return returnVertex;
}
