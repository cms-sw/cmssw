#include "TrackingTools/GsfTools/interface/MultiTrajectoryStateTransform.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/TrackReco/interface/TrackExtraFwd.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrackExtraFwd.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
#include "TrackingTools/GsfTools/interface/BasicMultiTrajectoryState.h"
#include "TrackingTools/TrajectoryParametrization/interface/LocalTrajectoryParameters.h"
#include "TrackingTools/TrajectoryParametrization/interface/LocalTrajectoryError.h"
#include "Geometry/CommonDetUnit/interface/TrackingGeometry.h"
#include "Geometry/CommonDetUnit/interface/GeomDet.h"
#include "TrackingTools/GeomPropagators/interface/AnalyticalPropagator.h"
#include "TrackingTools/GsfTools/interface/GsfPropagatorAdapter.h"
#include "TrackingTools/PatternTools/interface/TransverseImpactPointExtrapolator.h"
#include "TrackingTools/GsfTools/interface/MultiTrajectoryStateMode.h"

MultiTrajectoryStateTransform::~MultiTrajectoryStateTransform() { delete extrapolator_; }

TrajectoryStateOnSurface MultiTrajectoryStateTransform::outerStateOnSurface(const reco::GsfTrack& tk) const {
  return checkGeometry() ? outerStateOnSurface(tk, *geometry_, field_) : TrajectoryStateOnSurface();
}

TrajectoryStateOnSurface MultiTrajectoryStateTransform::innerStateOnSurface(const reco::GsfTrack& tk) const {
  return checkGeometry() ? innerStateOnSurface(tk, *geometry_, field_) : TrajectoryStateOnSurface();
}

bool MultiTrajectoryStateTransform::outerMomentumFromMode(const reco::GsfTrack& tk, GlobalVector& momentum) const {
  return multiTrajectoryStateMode::momentumFromModeCartesian(outerStateOnSurface(tk), momentum);
}

bool MultiTrajectoryStateTransform::innerMomentumFromMode(const reco::GsfTrack& tk, GlobalVector& momentum) const {
  return multiTrajectoryStateMode::momentumFromModeCartesian(outerStateOnSurface(tk), momentum);
}

TrajectoryStateOnSurface MultiTrajectoryStateTransform::outerStateOnSurface(const reco::GsfTrack& tk,
                                                                            const TrackingGeometry& geom,
                                                                            const MagneticField* field) {
  const Surface& surface = geom.idToDet(DetId(tk.extra()->outerDetId()))->surface();

  const reco::GsfTrackExtraRef& extra(tk.gsfExtra());
  return stateOnSurface(extra->outerStateWeights(),
                        extra->outerStateLocalParameters(),
                        extra->outerStateCovariances(),
                        extra->outerStateLocalPzSign(),
                        surface,
                        field);
}

TrajectoryStateOnSurface MultiTrajectoryStateTransform::innerStateOnSurface(const reco::GsfTrack& tk,
                                                                            const TrackingGeometry& geom,
                                                                            const MagneticField* field) {
  const Surface& surface = geom.idToDet(DetId(tk.extra()->innerDetId()))->surface();

  const reco::GsfTrackExtraRef& extra(tk.gsfExtra());
  return stateOnSurface(extra->innerStateWeights(),
                        extra->innerStateLocalParameters(),
                        extra->innerStateCovariances(),
                        extra->innerStateLocalPzSign(),
                        surface,
                        field);
}

TrajectoryStateOnSurface MultiTrajectoryStateTransform::stateOnSurface(const std::vector<double>& weights,
                                                                       const std::vector<ParameterVector>& parameters,
                                                                       const std::vector<CovarianceMatrix>& covariances,
                                                                       const double& pzSign,
                                                                       const Surface& surface,
                                                                       const MagneticField* field) {
  if (weights.empty())
    return TrajectoryStateOnSurface();

  unsigned int nc(weights.size());
  AlgebraicVector5 pars;
  AlgebraicSymMatrix55 cov;

  std::vector<TrajectoryStateOnSurface> components;
  components.reserve(nc);

  // create components TSOSs
  for (unsigned int i = 0; i < nc; i++) {
    // convert parameter vector and covariance matrix
    for (unsigned int j1 = 0; j1 < dimension; j1++) {
      pars[j1] = parameters[i](j1);
      for (unsigned int j2 = 0; j2 <= j1; j2++)
        cov(j1, j2) = covariances[i](j1, j2);  //FIXME: SMatrix copy constructor should handle this!!
    }
    // create local parameters & errors
    LocalTrajectoryParameters lp(pars, pzSign);
    LocalTrajectoryError le(cov);
    // create component
    components.push_back(TrajectoryStateOnSurface(weights[i], lp, le, surface, field));
  }
  return TrajectoryStateOnSurface((BasicTrajectoryState*)(new BasicMultiTrajectoryState(components)));
}

bool MultiTrajectoryStateTransform::checkGeometry() const {
  if (geometry_ && field_)
    return true;

  edm::LogError("MultiTrajectoryStateTransform") << "Missing ES components";
  return false;
}

TrajectoryStateOnSurface MultiTrajectoryStateTransform::extrapolatedState(const TrajectoryStateOnSurface tsos,
                                                                          const GlobalPoint& point) const {
  return checkExtrapolator() ? extrapolator_->extrapolate(tsos, point) : TrajectoryStateOnSurface();
}

bool MultiTrajectoryStateTransform::checkExtrapolator() const {
  if (extrapolator_)
    return true;

  if (field_ == nullptr) {
    edm::LogError("MultiTrajectoryStateTransform") << "Missing magnetic field";
    return false;
  }

  GsfPropagatorAdapter gsfPropagator(AnalyticalPropagator(field_, anyDirection));
  extrapolator_ = new TransverseImpactPointExtrapolator(gsfPropagator);
  return true;
}
