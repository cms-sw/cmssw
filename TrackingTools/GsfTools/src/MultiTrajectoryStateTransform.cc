#include "TrackingTools/GsfTools/interface/MultiTrajectoryStateTransform.h"
#include "DataFormats/TrackReco/interface/TrackExtraFwd.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrackExtraFwd.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
#include "TrackingTools/GsfTools/interface/BasicMultiTrajectoryState.h"
#include "TrackingTools/TrajectoryParametrization/interface/LocalTrajectoryParameters.h"
#include "TrackingTools/TrajectoryParametrization/interface/LocalTrajectoryError.h"
#include "Geometry/CommonDetUnit/interface/TrackingGeometry.h"
#include "Geometry/CommonDetUnit/interface/GeomDet.h"


TrajectoryStateOnSurface 
MultiTrajectoryStateTransform::outerStateOnSurface( const reco::GsfTrack& tk, 
						    const TrackingGeometry& geom,
						    const MagneticField* field) const
{
  const Surface& surface = geom.idToDet( DetId( tk.extra()->outerDetId()))->surface();

  const reco::GsfTrackExtraRef& extra(tk.gsfExtra());
  return stateOnSurface(extra->outerStateWeights(),
			extra->outerStateLocalParameters(),
			extra->outerStateCovariances(),
			extra->outerStateLocalPzSign(),
			surface,field);
}

TrajectoryStateOnSurface 
MultiTrajectoryStateTransform::innerStateOnSurface( const reco::GsfTrack& tk, 
						    const TrackingGeometry& geom,
						    const MagneticField* field) const
{
  const Surface& surface = geom.idToDet( DetId( tk.extra()->innerDetId()))->surface();

  const reco::GsfTrackExtraRef& extra(tk.gsfExtra());
  return stateOnSurface(extra->innerStateWeights(),
			extra->innerStateLocalParameters(),
			extra->innerStateCovariances(),
			extra->innerStateLocalPzSign(),
			surface,field);
}

TrajectoryStateOnSurface 
MultiTrajectoryStateTransform::stateOnSurface (const std::vector<double>& weights,
					       const std::vector<ParameterVector>& parameters,
					       const std::vector<CovarianceMatrix>& covariances,
					       const double& pzSign,
					       const Surface& surface,
					       const MagneticField* field) const
{
  if ( weights.empty() )  return TrajectoryStateOnSurface();
  
  unsigned int nc(weights.size());
  AlgebraicVector pars(dimension);
  AlgebraicSymMatrix cov(dimension);
  
  std::vector<TrajectoryStateOnSurface> components;
  components.reserve(nc);
  
  // create components TSOSs
  for ( unsigned int i=0; i<nc; i++ ) {
    // convert parameter vector and covariance matrix
    for ( unsigned int j1=0; j1<dimension; j1++ ) {
      pars[j1] = parameters[i](j1);
      for ( unsigned int j2=0; j2<=j1; j2++ ) 
	cov[j1][j2] = covariances[i](j1,j2);
    }
    // create local parameters & errors
    LocalTrajectoryParameters lp(pars,pzSign);
    LocalTrajectoryError le(cov);
    // create component
    components.push_back(TrajectoryStateOnSurface(lp,le,surface,field,weights[i]));
  }
  return 
    TrajectoryStateOnSurface(new BasicMultiTrajectoryState(components));
}
