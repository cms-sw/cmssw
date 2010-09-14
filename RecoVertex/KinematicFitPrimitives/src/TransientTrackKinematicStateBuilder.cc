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
                        const ParticleMass& mass,float m_sigma) const
{
//building initial kinematic state 
 return buildState(state,mass,m_sigma); 
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
using namespace std;
KinematicState
TransientTrackKinematicStateBuilder::buildState(const FreeTrajectoryState & state, 
	const ParticleMass& mass, float m_sigma) const
{ 
 AlgebraicVector7 par;
 AlgebraicSymMatrix77 cov;
 par(0) = state.position().x();
 par(1) = state.position().y();
 par(2) = state.position().z();
      
//getting the state of TransientTrack at the point
 par(3) = state.momentum().x();
 par(4) = state.momentum().y();
 par(5) = state.momentum().z();
 par(6) = mass;

//cartesian covariance matrix (x,y,z,p_x,p_y,p_z)
//and mass-related components stays unchanged
// if(!state.hasCartesianError()) throw VertexException("KinematicStateClosestToPointBuilder:: FTS passed has no error matrix!");

  FreeTrajectoryState curvFts(state.parameters(), state.curvilinearError());

//   cout <<"Transformation\n"<<curvFts.cartesianError().matrix()<<endl;
 cov.Place_at(curvFts.cartesianError().matrix(),0,0);
 cov(6,6) = m_sigma * m_sigma;

//making parameters & error
 KinematicParameters wPar(par);
 KinematicParametersError wEr(cov);
 return KinematicState(wPar,wEr,state.charge(), &state.parameters().magneticField());
}
