#include "RecoVertex/KinematicFitPrimitives/interface/TransientTrackKinematicStateBuilder.h"

using namespace reco;

KinematicState TransientTrackKinematicStateBuilder::operator()(const TransientTrack& track,
                                                               const ParticleMass& m,
                                                               float m_sigma) const {
  // FreeTrajectoryState * recState = track.impactPointState().freeState();
  return buildState(*(track.impactPointState().freeState()), m, m_sigma);
}

//KinematicState
//TransientTrackKinematicStateBuilder::operator()(const KinematicParameters& par,
//	const KinematicParametersError& er, const TrackCharge& ch,
//	const MagneticField* field) const
//{
//  return KinematicState(par, er, ch, field);
//}

KinematicState TransientTrackKinematicStateBuilder::operator()(const TransientTrack& track,
                                                               const GlobalPoint& point,
                                                               const ParticleMass& m,
                                                               float m_sigma) const {
  //  FreeTrajectoryState  recState = track.trajectoryStateClosestToPoint(point).theState();
  return buildState(track.trajectoryStateClosestToPoint(point).theState(), m, m_sigma);
}

KinematicState TransientTrackKinematicStateBuilder::operator()(const FreeTrajectoryState& state,
                                                               const ParticleMass& mass,
                                                               float m_sigma) const {
  //building initial kinematic state
  return buildState(state, mass, m_sigma);
}

KinematicState TransientTrackKinematicStateBuilder::operator()(const FreeTrajectoryState& state,
                                                               const ParticleMass& mass,
                                                               float m_sigma,
                                                               const GlobalPoint& point) const {
  //building initial kinematic state
  KinematicState res = buildState(state, mass, m_sigma);

  //and propagating it to given point if needed
  GlobalPoint inPos = state.position();
  if ((inPos.x() != point.x()) || (inPos.y() != point.y()) || (inPos.z() != point.z())) {
    res = propagator.propagateToTheTransversePCA(res, point);
  }
  return res;
}

PerigeeKinematicState TransientTrackKinematicStateBuilder::operator()(const KinematicState& state,
                                                                      const GlobalPoint& point) const {
  KinematicState nState = propagator.propagateToTheTransversePCA(state, point);
  return PerigeeKinematicState(nState, point);
}

KinematicState TransientTrackKinematicStateBuilder::buildState(const FreeTrajectoryState& state,
                                                               const ParticleMass& mass,
                                                               float m_sigma) const {
  return KinematicState(state, mass, m_sigma);
}
