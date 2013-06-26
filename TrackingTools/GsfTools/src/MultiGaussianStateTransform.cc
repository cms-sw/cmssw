#include "TrackingTools/GsfTools/interface/MultiGaussianStateTransform.h"
#include "TrackingTools/GsfTools/interface/MultiGaussianState1D.h"
#include "DataFormats/TrackReco/interface/TrackExtraFwd.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrackExtraFwd.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
#include "FWCore/Utilities/interface/Exception.h"

MultiGaussianState<MultiGaussianStateTransform::N>
MultiGaussianStateTransform::outerMultiState (const reco::GsfTrack& tk)
{
  const reco::GsfTrackExtraRef& extra(tk.gsfExtra());
  return multiState(extra->outerStateLocalParameters(),
		    extra->outerStateCovariances(),
		    extra->outerStateWeights());
}

MultiGaussianState<MultiGaussianStateTransform::N>
MultiGaussianStateTransform::innerMultiState (const reco::GsfTrack& tk)
{
  const reco::GsfTrackExtraRef& extra(tk.gsfExtra());
  return multiState(extra->innerStateLocalParameters(),
	       extra->innerStateCovariances(),
	       extra->innerStateWeights());
}

MultiGaussianState1D
MultiGaussianStateTransform::outerMultiState1D (const reco::GsfTrack& tk,
						unsigned int index)
{
  if ( index>=N )  
    throw cms::Exception("LogicError") << "MultiGaussianStateTransform: index out of range";

  const reco::GsfTrackExtraRef& extra(tk.gsfExtra());
  return multiState1D(extra->outerStateLocalParameters(),
		      extra->outerStateCovariances(),
		      extra->outerStateWeights(),
		      index);
}

MultiGaussianState1D
MultiGaussianStateTransform::innerMultiState1D (const reco::GsfTrack& tk,
						unsigned int index)
{
  if ( index>=N )  
    throw cms::Exception("LogicError") << "MultiGaussianStateTransform: index out of range";

  const reco::GsfTrackExtraRef& extra(tk.gsfExtra());
  return multiState1D(extra->innerStateLocalParameters(),
		      extra->innerStateCovariances(),
		      extra->innerStateWeights(),
		      index);
}

MultiGaussianState<MultiGaussianStateTransform::N> 
MultiGaussianStateTransform::multiState (const std::vector<MultiGaussianState<N>::Vector>& parameters,
					 const std::vector<MultiGaussianState<N>::Matrix>& covariances,
					 const std::vector<double>& weights)
{
  unsigned int nc = parameters.size();
  MultiGaussianState<N>::SingleStateContainer components;
  components.reserve(nc);
  for ( unsigned int i=0; i<nc; ++i ) {
    MultiGaussianState<N>::SingleStatePtr 
      sgs(new MultiGaussianState<N>::SingleState(parameters[i],covariances[i],weights[i]));
    components.push_back(sgs);
  }
  return MultiGaussianState<N>(components);
}

MultiGaussianState1D
MultiGaussianStateTransform::multiState1D (const std::vector<MultiGaussianState<N>::Vector>& parameters,
					   const std::vector<MultiGaussianState<N>::Matrix>& covariances,
					   const std::vector<double>& weights, unsigned int index)
{
  unsigned int nc = parameters.size();
  MultiGaussianState1D::SingleState1dContainer components;
  components.reserve(nc);
  for ( unsigned int i=0; i<nc; ++i ) {
    components.push_back(SingleGaussianState1D(parameters[i](index),
					       covariances[i](index,index),
					       weights[i]));
  }
  return MultiGaussianState1D(components);
}

MultiGaussianState<5> 
MultiGaussianStateTransform::multiState (const TrajectoryStateOnSurface tsos)
{
  std::vector<TrajectoryStateOnSurface> tsosComponents(tsos.components());
  MultiGaussianState<5>::SingleStateContainer components;
  components.reserve(tsosComponents.size());
  for ( std::vector<TrajectoryStateOnSurface>::const_iterator i=tsosComponents.begin();
	i!=tsosComponents.end(); ++i ) {
    MultiGaussianState<5>::SingleStatePtr 
      sgs(new MultiGaussianState<5>::SingleState(i->localParameters().vector(),
						 i->localError().matrix(),
						 i->weight()));
    components.push_back(sgs);
  }
  return MultiGaussianState<5>(components);
}

MultiGaussianState1D
MultiGaussianStateTransform::multiState1D (const TrajectoryStateOnSurface tsos,
					   unsigned int index)
{
  if ( index>=N )  
    throw cms::Exception("LogicError") << "MultiGaussianStateTransform: index out of range";
  std::vector<TrajectoryStateOnSurface> tsosComponents(tsos.components());
  MultiGaussianState1D::SingleState1dContainer components;
  components.reserve(tsosComponents.size());
  for ( std::vector<TrajectoryStateOnSurface>::const_iterator i=tsosComponents.begin();
	i!=tsosComponents.end(); ++i ) {
    components.push_back(SingleGaussianState1D(i->localParameters().vector()(index),
					       i->localError().matrix()(index,index),
					       i->weight()));
  }
  return MultiGaussianState1D(components);
}

TrajectoryStateOnSurface
MultiGaussianStateTransform::tsosFromSingleState (const SingleGaussianState<5>& singleState,
						  const TrajectoryStateOnSurface refTsos)
{
  const LocalTrajectoryParameters& refPars(refTsos.localParameters());
  double pzSign = refPars.pzSign();
  bool charged = refPars.charge()!=0;
  LocalTrajectoryParameters pars(singleState.mean(),pzSign,charged);
  LocalTrajectoryError errs(singleState.covariance());
  // return state (doesn't use weight of the single state)
  return TrajectoryStateOnSurface(pars,errs,refTsos.surface(),refTsos.magneticField());
}
