#include "TrackingTools/MaterialEffects/interface/VolumeMaterialEffectsUpdator.h"

#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
#include "DataFormats/TrajectorySeed/interface/PropagationDirection.h"
#include "TrackingTools/MaterialEffects/interface/VolumeMaterialEffectsEstimate.h"

#include "DataFormats/GeometryCommonDetAlgo/interface/PerpendicularBoundPlaneBuilder.h"
#include "TrackingTools/AnalyticalJacobians/interface/JacobianCurvilinearToLocal.h"
#include "TrackingTools/AnalyticalJacobians/interface/JacobianLocalToCurvilinear.h"

TrajectoryStateOnSurface
VolumeMaterialEffectsUpdator::updateState (const TrajectoryStateOnSurface& tsos, 
					   const PropagationDirection propDir,
					   const Estimate& estimate) const
{
  return updateState(tsos,propDir,EstimateContainer(1,&estimate));
}

TrajectoryStateOnSurface
VolumeMaterialEffectsUpdator::updateState (const TrajectoryStateOnSurface& tsos, 
					   const PropagationDirection propDir,
					   const EstimateContainer& estimates) const
{
  // Sanity check on propagation direction
  if ( propDir==anyDirection )  return TrajectoryStateOnSurface();
  //
  // Update momentum. In case of failure: return invalid state
  //
  double dpSum(0.);
  for ( EstimateContainer::const_iterator i=estimates.begin();
	i!=estimates.end(); ++i ) {
    double dp = (**i).deltaP();
    if ( propDir==alongMomentum )  dpSum += dp;
    else  dpSum -= dp;
  }
  LocalTrajectoryParameters lp = tsos.localParameters();
  if ( !lp.updateP(dpSum) )  return TrajectoryStateOnSurface();
  //
  // Update covariance matrix?
  //
  if ( tsos.hasError() ) {
//     AlgebraicSymMatrix55 eloc(tsos.localError().matrix());
    AlgebraicSymMatrix55 matCov;
    for ( EstimateContainer::const_iterator i=estimates.begin();
	  i!=estimates.end(); ++i ) {
      matCov += (**i).deltaLocalError();
    }
    //
    // transform to local system of trackfor the time being: brute force
    // could / should try to construct the matrix in curvilinear
    //
    // Plane consistent with material effects estimate (perp. to track at ref. pos.)
    ReferenceCountingPointer<BoundPlane> 
      perpPlane(PerpendicularBoundPlaneBuilder()(tsos.globalPosition(),
						 tsos.globalMomentum()));
    // Parameters need to construct Jacobian (ref. point and ref. direction)
    LocalTrajectoryParameters perpPars(LocalPoint(0.,0.,0.),
				       LocalVector(0.,0.,1.),
				       tsos.charge());
    // Jacobian to go from perpendicular plane to curvilinear
    JacobianLocalToCurvilinear jacLocToCurv(*perpPlane,perpPars,*tsos.magneticField());
    // Jacobian to go from curvilinear to local frame of the reference tsos
    JacobianCurvilinearToLocal jacCurvToLoc(tsos.surface(),tsos.localParameters(),
					    *tsos.magneticField());
    // Combined Jacobian
    AlgebraicMatrix55 jac(jacLocToCurv.jacobian()*jacCurvToLoc.jacobian());
    // Add transformed material effects error matrix to the one from the TSOS
    AlgebraicSymMatrix55 eloc(tsos.localError().matrix());
    eloc += ROOT::Math::Similarity(jac,matCov);
    return TrajectoryStateOnSurface(lp,LocalTrajectoryError(eloc),tsos.surface(),
				    &(tsos.globalParameters().magneticField()),
				    tsos.surfaceSide());
  }
  else {
    return TrajectoryStateOnSurface(lp,tsos.surface(),
				    &(tsos.globalParameters().magneticField()),
				    tsos.surfaceSide());
  }
}
