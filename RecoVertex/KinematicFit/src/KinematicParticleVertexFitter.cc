#include "RecoVertex/KinematicFit/interface/KinematicParticleVertexFitter.h"
// #include "Vertex/LinearizationPointFinders/interface/LMSLinearizationPointFinder.h"
#include "RecoVertex/KinematicFit/interface/FinalTreeBuilder.h"
// #include "RecoVertex/VertexTools/interface/SequentialVertexSmoother.h"
// #include "RecoVertex/KalmanVertexFit/interface/KalmanVertexUpdator.h"
// #include "RecoVertex/KalmanVertexFit/interface/KalmanVertexTrackUpdator.h"
// #include "RecoVertex/KalmanVertexFit/interface/KalmanSmoothedVertexChi2Estimator.h"
// #include "RecoVertex/KalmanVertexFit/interface/KalmanTrackToTrackCovCalculator.h"
#include "RecoVertex/VertexPrimitives/interface/VertexException.h"
#include "RecoVertex/LinearizationPointFinders/interface/DefaultLinearizationPointFinder.h"

KinematicParticleVertexFitter::KinematicParticleVertexFitter()
{ 
  fitter = new KalmanVertexFitter(true);
//FIXME
 pointFinder =  new DefaultLinearizationPointFinder();
 vFactory = new VertexTrackFactory();
}

// KinematicParticleVertexFitter::KinematicParticleVertexFitter(const LinearizationPointFinder&  finder)
// { 
//  pointFinder = finder.clone();
//  vFactory = new VertexTrackFactory();
//  KalmanVertexTrackUpdator kvtu;
//  KalmanSmoothedVertexChi2Estimator est;
//  KalmanTrackToTrackCovCalculator cl;
//  SequentialVertexSmoother smoother(kvtu,est,cl);
//  KalmanVertexUpdator updator; 
//  fitter = new SequentialKinematicVertexFitter(updator, smoother);
// }

KinematicParticleVertexFitter::~KinematicParticleVertexFitter()
{
 delete vFactory;
 delete pointFinder;
 delete fitter;
}
 
RefCountedKinematicTree KinematicParticleVertexFitter::fit(vector<RefCountedKinematicParticle> particles) const
{
//sorting the input 
 if(particles.size()<2) throw VertexException("KinematicParticleVertexFitter::input states are less than 2"); 
 InputSort iSort;
 pair<vector<RefCountedKinematicParticle>, vector<FreeTrajectoryState> > input = iSort.sort(particles);
 vector<RefCountedKinematicParticle> & newPart = input.first;
 vector<FreeTrajectoryState> & freeStates = input.second;

 GlobalPoint linPoint = pointFinder->getLinearizationPoint(freeStates);
  
// cout<<"Linearization point found"<<endl; 
//creating VertexTracks to make vertex  
 vector<RefCountedVertexTrack> vTracks;
 
//making initial veretx seed with lin point as position and a fake error
 AlgebraicSymMatrix we(3,1);
 GlobalError error(we*10000);
 VertexState state(linPoint, error);
 
//vector of Vertex Tracks to fit
 vector<RefCountedVertexTrack> ttf; 
 for(vector<RefCountedKinematicParticle>::const_iterator i = newPart.begin();i != newPart.end();i++)
 {ttf.push_back(vFactory->vertexTrack((*i)->particleLinearizedTrackState(linPoint),state,1.));}

// //debugging code to check neutrals: 
//  for(vector<RefCountedVertexTrack>::const_iterator i = ttf.begin(); i!=ttf.end(); i++)
//  {
// //   cout<<"predicted state momentum error"<<(*i)->linearizedTrack()->predictedStateMomentumError()<<endl;
// //  cout<<"Momentum jacobian"<<(*i)->linearizedTrack()->momentumJacobian() <<endl;
//  //  cout<<"predicted state momentum "<<(*i)->linearizedTrack()->predictedStateMomentum()<<endl;
// //   cout<<"constant term"<<(*i)->linearizedTrack()->constantTerm()<<endl;
// 
//  }

 CachingVertex vtx = fitter->vertex(ttf); 

 FinalTreeBuilder tBuilder;
 return tBuilder.buildTree(vtx, newPart); 
}
