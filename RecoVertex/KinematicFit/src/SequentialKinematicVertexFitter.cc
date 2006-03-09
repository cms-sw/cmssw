#include "RecoVertex/KinematicFit/interface/SequentialKinematicVertexFitter.h"
// #include "Utilities/UI/interface/SimpleConfigurable.h"

SequentialKinematicVertexFitter::SequentialKinematicVertexFitter(const VertexUpdator& theUpdator, 
                                                           const VertexSmoother&  theSmoother)
{
 smoother = theSmoother.clone();
 updator = theUpdator.clone();
 readParameters();
}

SequentialKinematicVertexFitter::~SequentialKinematicVertexFitter()
{
 delete updator;
 delete smoother;
}

CachingVertex SequentialKinematicVertexFitter::vertex
                   (const vector<RefCountedVertexTrack> & particles) const
{
 GlobalPoint linP = particles[1]->linearizedTrack()->linearizationPoint(); 
 AlgebraicSymMatrix we(3,1);
 GlobalError error(we*10000);
 VertexState state(linP, error);
 return fit(particles, state, false);
}					  
						  										      
vector<RefCountedVertexTrack> SequentialKinematicVertexFitter::reLinearizeTracks
 (const vector<RefCountedVertexTrack> & tracks, const VertexState seed) const
{
 GlobalPoint linP = seed.position();
 vector<RefCountedVertexTrack> finalTracks;
 for(vector<RefCountedVertexTrack>::const_iterator i = tracks.begin(); i != tracks.end(); i++)
 {
 
//fist checking that we are given with real kinematic linearized tracks
//and not something esle  
  LinearizedTrackState  * r_s = &(*(*i)->linearizedTrack());
  RefCountedLinearizedTrackState lTrData = r_s->stateWithNewLinearizationPoint(linP);
  RefCountedVertexTrack vTrData = vTrackFactory.vertexTrack(lTrData,seed);
  finalTracks.push_back(vTrData); 
 }
 return finalTracks;
}


CachingVertex  SequentialKinematicVertexFitter::fit(const vector<RefCountedVertexTrack> & tracks,
			const VertexState priorSeed, bool withPrior) const
{
 vector<RefCountedVertexTrack> initialTracks;
 GlobalPoint priorVertexPosition = priorSeed.position();
 GlobalError priorVertexError = priorSeed.error();
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
  do
  {
   CachingVertex fVertex = initialVertex;
   
// make new linearized and vertex tracks for the next iteration
   if(step != 0) globalVTracks = reLinearizeTracks(tracks, 
   					returnVertex.vertexState());

    // update sequentially the vertex estimate
    for(vector<RefCountedVertexTrack>::const_iterator i = globalVTracks.begin();
   	 i != globalVTracks.end(); i++)
    {  
      fVertex = updator->add(fVertex,*i);
    }
    previousPosition = newPosition;
    newPosition = fVertex.position();
 
    returnVertex = fVertex;
    globalVTracks.clear();
    step++;
  } while ( (step != theMaxStep) &&
  	    ((previousPosition - newPosition).transverse() > theMaxShift) );

//smoothing options
//  CachingVertex r_l = returnVertex;
  if (smoother != 0)
  {
    returnVertex = smoother->smooth(returnVertex);
//    r_l = smoother->smooth(returnVertex);
   }
//  return r_l;
  return returnVertex;
}							      
										      
void SequentialKinematicVertexFitter::readParameters()
{
//FIXME
//   static SimpleConfigurable<float>
//     maxShiftSimTrackConfigurable(0.1,"SequentialKinematicVertexFitter:maximumDistance");
//   theMaxShift = maxShiftSimTrackConfigurable.value();
// 
//   static SimpleConfigurable<int>
//     maxStepConfigurable(10,"SequentiaKinematiclVertexFitter:maximumNumberOfIterations");
//   theMaxStep = maxStepConfigurable.value();
}
								    
