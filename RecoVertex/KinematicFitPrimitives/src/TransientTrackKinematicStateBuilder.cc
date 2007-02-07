#include "RecoVertex/KinematicFitPrimitives/interface/TransientTrackKinematicStateBuilder.h"

using namespace reco;


KinematicState TransientTrackKinematicStateBuilder::operator()(const TransientTrack& track, 
                                         const ParticleMass& m, float m_sigma) const 
{ 
// FreeTrajectoryState * recState = track.impactPointState().freeState(); 
 return buildState(*(track.impactPointState().freeState()), m, m_sigma);
} 

KinematicState
TransientTrackKinematicStateBuilder::operator()(const KinematicParameters& par,
	const KinematicParametersError& er, const TrackCharge& ch,
	const MagneticField* field) const
{
  return KinematicState(par, er, ch, field);
}
 
KinematicState TransientTrackKinematicStateBuilder::operator()(const TransientTrack& track, 
                          const GlobalPoint& point, const ParticleMass& m,float m_sigma) const
{
//  FreeTrajectoryState  recState = track.trajectoryStateClosestToPoint(point).theState();
 return buildState( track.trajectoryStateClosestToPoint(point).theState(), m, m_sigma);
} 

KinematicState TransientTrackKinematicStateBuilder::operator()(const FreeTrajectoryState& state,
                        const ParticleMass& mass,float m_sigma, const GlobalPoint& point) const
{
//building initial kinematic state 
 KinematicState res = buildState(state,mass,m_sigma);
 
//and propagating it to given point if needed
 GlobalPoint inPos = state.position();
 if((inPos.x() != point.x())||(inPos.y() != point.y())||(inPos.z() != point.z()))
 {res = propagator.propagateToTheTransversePCA(res,point);}  
 return res; 
}
			    
PerigeeKinematicState TransientTrackKinematicStateBuilder::operator()(const KinematicState& state, 
                                                                  const GlobalPoint& point)const
{
 KinematicState nState = propagator.propagateToTheTransversePCA(state, point);
 return PerigeeKinematicState(nState, point);
}	

KinematicState
TransientTrackKinematicStateBuilder::buildState(const FreeTrajectoryState & state, 
	const ParticleMass& mass, float m_sigma) const
{ 
 AlgebraicVector par(7);
 AlgebraicSymMatrix cov(7,0);
 par(1) = state.position().x();
 par(2) = state.position().y();
 par(3) = state.position().z();
      
//getting the state of TransientTrack at the point
 par(4) = state.momentum().x();
 par(5) = state.momentum().y();
 par(6) = state.momentum().z();
 par(7) = mass;

//cartesian covariance matrix (x,y,z,p_x,p_y,p_z)
//and mass-related components stays unchanged
 if(!state.hasCartesianError()) throw VertexException("KinematicStateClosestToPointBuilder:: FTS passed has no error matrix!");
 AlgebraicSymMatrix  cartCov = state.cartesianError().matrix();
 cov.sub(1,cartCov);
 cov(7,7) = m_sigma * m_sigma;

//making parameters & error
 KinematicParameters wPar(par);
 KinematicParametersError wEr(cov);
 return KinematicState(wPar,wEr,state.charge(), &state.parameters().magneticField());
}
