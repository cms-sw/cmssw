#include "RecoVertex/KinematicFitPrimitives/interface/KinematicRefittedTrackState.h"
#include "TrackingTools/GeomPropagators/interface/AnalyticalPropagator.h"
#include "DataFormats/TrajectorySeed/interface/PropagationDirection.h"
#include "RecoVertex/KinematicFitPrimitives/interface/KinematicPerigeeConversions.h"


KinematicRefittedTrackState::KinematicRefittedTrackState(const KinematicState& st,
		const AlgebraicVector4& mv)		
{
  state = st;
  momentumAtVertex = mv;
}

AlgebraicVector6 KinematicRefittedTrackState::parameters() const
{
  KinematicPerigeeConversions conv;
  return conv.extendedPerigeeFromKinematicParameters(state,state.globalPosition()).vector();
}
  
AlgebraicSymMatrix66 KinematicRefittedTrackState::covariance() const 
{
  throw VertexException("KinematicRefittedTrackState::Fishy covariance called");
  return AlgebraicSymMatrix66();
}

AlgebraicVector7 KinematicRefittedTrackState::kinematicParameters() const
{return state.kinematicParameters().vector();}

AlgebraicSymMatrix77 KinematicRefittedTrackState::kinematicParametersCovariance() const
{return state.kinematicParametersError().matrix();}


FreeTrajectoryState KinematicRefittedTrackState::freeTrajectoryState() const
{
 return state.freeTrajectoryState();
}

GlobalPoint KinematicRefittedTrackState::position() const
{return state.globalPosition();}

AlgebraicVector4 KinematicRefittedTrackState::kinematicMomentumVector() const
{
 GlobalVector mm = state.globalMomentum();
 AlgebraicVector4 mr;
 mr[0] = mm.x();
 mr[1] = mm.y();
 mr[2] = mm.z();
 mr[3] = state.mass();
 return mr;
}

AlgebraicVector4 KinematicRefittedTrackState::momentumVector() const
{
 return momentumAtVertex;
}


TrajectoryStateOnSurface KinematicRefittedTrackState::trajectoryStateOnSurface(const Surface & surface) const
{
  AnalyticalPropagator thePropagator(state.magneticField(), anyDirection);
 return thePropagator.propagate(freeTrajectoryState(), surface);
}

TrajectoryStateOnSurface KinematicRefittedTrackState::trajectoryStateOnSurface(const Surface & surface, 
                                                   const Propagator & propagator) const
{
 std::unique_ptr<Propagator> thePropagator( propagator.clone());
 thePropagator->setPropagationDirection(anyDirection);
 return thePropagator->propagate(freeTrajectoryState(), surface);
}
						   
 double KinematicRefittedTrackState::weight() const
{ return 1.;}

ReferenceCountingPointer<RefittedTrackState<6> >  KinematicRefittedTrackState::stateWithNewWeight
  	(const double newWeight) const
{
 std::cout<<"WARNING: Change weight for Kinematic state called, weigt will stay to be equal 1."<<std::endl;
 return RefCountedRefittedTrackState( 
  				const_cast<KinematicRefittedTrackState*>(this));
}

std::vector< ReferenceCountingPointer<RefittedTrackState<6> > > KinematicRefittedTrackState::components() const
{
 std::vector<RefCountedRefittedTrackState> result; result.reserve(1);
 result.push_back(RefCountedRefittedTrackState( 
  				const_cast<KinematicRefittedTrackState*>(this)));
 return result;
}						   



reco::TransientTrack KinematicRefittedTrackState::transientTrack() const
{
  throw VertexException("KinematicRefittedTrackState::Can Not write a TransientTrack");
//  TransientTrackFromFTSFactory factory;
//   return factory.build(freeTrajectoryState());
}
