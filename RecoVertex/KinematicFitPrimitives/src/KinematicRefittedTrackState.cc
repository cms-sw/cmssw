#include "RecoVertex/KinematicFitPrimitives/interface/KinematicRefittedTrackState.h"
#include "RecoVertex/VertexPrimitives/interface/RefCountedRefittedTrackState.h"
#include "TrackingTools/GeomPropagators/interface/AnalyticalPropagator.h"
#include "TrackingTools/GeomPropagators/interface/PropagationDirection.h"
#include "RecoVertex/KinematicFitPrimitives/interface/KinematicPerigeeConversions.h"
#include "TrackingTools/TrajectoryState/interface/FakeField.h"


KinematicRefittedTrackState::KinematicRefittedTrackState(const KinematicState& st)		
{state = st;}

AlgebraicVector KinematicRefittedTrackState::parameters() const
{return state.kinematicParameters().vector();}
  
AlgebraicSymMatrix KinematicRefittedTrackState::covariance() const 
{return state.kinematicParametersError().matrix();}

FreeTrajectoryState KinematicRefittedTrackState::freeTrajectoryState() const
{
 return state.freeTrajectoryState();
}

GlobalPoint KinematicRefittedTrackState::position() const
{return state.globalPosition();}

AlgebraicVector KinematicRefittedTrackState::kinematicMomentumVector() const
{
 GlobalVector mm = state.globalMomentum();
 AlgebraicVector mr(4);
 mr(1) = mm.x();
 mr(2) = mm.y();
 mr(3) = mm.z();
 mr(4) = state.mass();
 return mr;
}

AlgebraicVector KinematicRefittedTrackState::momentumVector() const
{

 KinematicPerigeeConversions conv;
 
 ExtendedPerigeeTrajectoryParameters pState = 
                conv.extendedPerigeeFromKinematicParameters(state,state.globalPosition());

 AlgebraicVector mr(4);
 mr(4) = pState.vector()(6);
 mr(1) = pState.vector()(1);
 mr(2) = pState.vector()(2);
 mr(3) = pState.vector()(3);
 return mr;
}


TrajectoryStateOnSurface KinematicRefittedTrackState::trajectoryStateOnSurface(const Surface & surface) const
{
 AnalyticalPropagator thePropagator(TrackingTools::FakeField::Field::field(), 
 					anyDirection);
 return thePropagator.propagate(freeTrajectoryState(), surface);
}

TrajectoryStateOnSurface KinematicRefittedTrackState::trajectoryStateOnSurface(const Surface & surface, 
                                                   const Propagator & propagator) const
{
 std::auto_ptr<Propagator> thePropagator( propagator.clone());
 thePropagator->setPropagationDirection(anyDirection);
 return thePropagator->propagate(freeTrajectoryState(), surface);
}
						   
 double KinematicRefittedTrackState::weight() const
{ return 1.;}

ReferenceCountingPointer<RefittedTrackState>  KinematicRefittedTrackState::stateWithNewWeight
  	(const double newWeight) const
{
 std::cout<<"WARNING: Change weight for Kinematic state called, weigt will stay to be equal 1."<<std::endl;
 return RefCountedRefittedTrackState( 
  				const_cast<KinematicRefittedTrackState*>(this));
}

std::vector< ReferenceCountingPointer<RefittedTrackState> > KinematicRefittedTrackState::components() const
{
 std::vector<RefCountedRefittedTrackState> result; result.reserve(1);
 result.push_back(RefCountedRefittedTrackState( 
  				const_cast<KinematicRefittedTrackState*>(this)));
 return result;
}						   



