#include "RecoVertex/KinematicFitPrimitives/interface/VirtualKinematicParticle.h"
#include "TrackingTools/TrajectoryState/interface/FreeTrajectoryState.h"

VirtualKinematicParticle::VirtualKinematicParticle
	(const KinematicState& kineState, float& chiSquared, float& degreesOfFr,
	 KinematicConstraint * lastConstraint,
	 ReferenceCountingPointer<KinematicParticle> previousParticle,
	 KinematicStatePropagator * pr)
{
  theField = kineState.magneticField();
 if(previousParticle.get() == 0)
 {
  initState = kineState;
 }else{initState = previousParticle->initialState();}
 cState = kineState;
 pState = previousParticle;
 chi2 = chiSquared;
 ndf = degreesOfFr;
 lConstraint = lastConstraint;
 if(pr != 0)
 {
  propagator = pr->clone();
 }else{
  propagator = new TrackKinematicStatePropagator();
 }
 tree = 0;
}

VirtualKinematicParticle::~VirtualKinematicParticle()
{delete propagator;}

bool VirtualKinematicParticle::operator==(const KinematicParticle& other)const
{
 bool dc = false;

//first looking if this is an object of the same type
 const  KinematicParticle * lp = &other;
 const VirtualKinematicParticle * lPart = dynamic_cast<const VirtualKinematicParticle * >(lp);
 if(lPart != 0 && initialState() == lPart->initialState()) dc = true;
 return dc;
}

bool VirtualKinematicParticle::operator==(const ReferenceCountingPointer<KinematicParticle>& other) const
{
 bool res = false;
 if(*this == *other) res = true;
 return res;
}

bool VirtualKinematicParticle::operator!=(const KinematicParticle& other)const
{
 if (*this == other){
  return false;
 }else{return true;}
}

KinematicState VirtualKinematicParticle::stateAtPoint(const GlobalPoint& point)const
{
 GlobalPoint iP = cState.kinematicParameters().position();
 if((iP.x() == point.x())&&(iP.y() == point.y())&&(iP.z() == point.z()))
 {
  return cState ;
 }else{return propagator->propagateToTheTransversePCA(cState,point);} }

RefCountedKinematicParticle VirtualKinematicParticle::refittedParticle(const KinematicState& state,
                                                       float chi2, float ndf, KinematicConstraint * cons)const
{
 VirtualKinematicParticle * ncp = const_cast<VirtualKinematicParticle * >(this);
 ReferenceCountingPointer<KinematicParticle> current = ReferenceCountingPointer<KinematicParticle>(ncp);
 return ReferenceCountingPointer<KinematicParticle>(new VirtualKinematicParticle(state,chi2,ndf,cons,current,
                                                                                                propagator));
}

VirtualKinematicParticle::RefCountedLinearizedTrackState
VirtualKinematicParticle::particleLinearizedTrackState(const GlobalPoint& point)const
{
 VirtualKinematicParticle * cr = const_cast<VirtualKinematicParticle * >(this);
 RefCountedKinematicParticle lp = ReferenceCountingPointer<KinematicParticle>(cr);
 return linFactory.linearizedTrackState(point,lp);
}
