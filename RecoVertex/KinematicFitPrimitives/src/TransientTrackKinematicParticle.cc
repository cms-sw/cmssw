#include "RecoVertex/KinematicFitPrimitives/interface/TransientTrackKinematicParticle.h"
#include "TrackingTools/TrajectoryState/interface/FreeTrajectoryState.h"

using namespace reco;

TransientTrackKinematicParticle::TransientTrackKinematicParticle
	(const KinematicState& kineState, float& chiSquared,
	float& degreesOfFr, KinematicConstraint * lastConstraint,
	ReferenceCountingPointer<KinematicParticle> previousParticle,
	KinematicStatePropagator * pr, const TransientTrack * initialTrack)
{
  theField = kineState.magneticField();
 if(previousParticle.get() == 0)
 { 
  initState = kineState;
}else{initState = previousParticle->initialState();}
 cState = kineState;
 inTrack = initialTrack;
 pState = previousParticle;
 chi2 = chiSquared;
 ndf = degreesOfFr;
 lConstraint = lastConstraint;
 if(pr!=0)
 {
  propagator = pr->clone();
 }else{
  propagator = new TrackKinematicStatePropagator();
 }
 tree = 0;
} 

TransientTrackKinematicParticle::~TransientTrackKinematicParticle()
{delete propagator;}
 
bool TransientTrackKinematicParticle::operator==(const KinematicParticle& other)const
{
 bool dc = false;
 
//first looking if this is an object of the same type
 const  KinematicParticle * lp = &other;
 const TransientTrackKinematicParticle * lPart = dynamic_cast<const TransientTrackKinematicParticle * >(lp);
 if(lPart != 0){
 
//then comparing particle with their initial TransientTracks 
  if((initialTransientTrack())&&(lPart->initialTransientTrack()))
  {
   if(initialTransientTrack() == lPart->initialTransientTrack()) dc = true;
  }else{if(initialState() == lPart->initialState()) dc = true;} 
 }
 return dc; 
}

bool TransientTrackKinematicParticle::operator==(const ReferenceCountingPointer<KinematicParticle>& other) const
{
 bool res = false;
 if(*this == *other) res = true;
 return res;
}

bool  TransientTrackKinematicParticle::operator!=(const KinematicParticle& other)const
{
 if (*this == other){
  return false;
 }else{return true;} 
}

KinematicState TransientTrackKinematicParticle::stateAtPoint(const GlobalPoint& point)const
{
 GlobalPoint iP = cState.kinematicParameters().position();
 if((iP.x() == point.x())&&(iP.y() == point.y())&&(iP.z() == point.z()))
 {
  return cState ;
 }else{return propagator->propagateToTheTransversePCA(cState,point);} 
}

//FreeTrajectoryState TransientTrackKinematicParticle::initialStateFTS() const
//{return initState.freeTrajectoryState();}

const TransientTrack * TransientTrackKinematicParticle::initialTransientTrack() const
{return inTrack;}  

ReferenceCountingPointer<KinematicParticle> TransientTrackKinematicParticle::refittedParticle(const KinematicState& state,
                                                             float chi2, float ndf, KinematicConstraint * cons)const
{
 TransientTrackKinematicParticle * ncp = const_cast<TransientTrackKinematicParticle * >(this);
 return ReferenceCountingPointer<KinematicParticle>(new TransientTrackKinematicParticle(state,chi2,ndf,cons,
                          ReferenceCountingPointer<KinematicParticle>(ncp), propagator, initialTransientTrack()));
}

TransientTrackKinematicParticle::RefCountedLinearizedTrackState
TransientTrackKinematicParticle::particleLinearizedTrackState(const GlobalPoint& point)const	
{
 TransientTrackKinematicParticle * cr = const_cast<TransientTrackKinematicParticle * >(this);
 RefCountedKinematicParticle lp = ReferenceCountingPointer<KinematicParticle>(cr);
 return linFactory.linearizedTrackState(point,lp);
}		      
							
