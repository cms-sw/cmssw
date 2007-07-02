#include "TrackingTools/GsfTools/interface/MultiGaussianStateTransform.h"
#include "DataFormats/TrackReco/interface/TrackExtraFwd.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrackExtraFwd.h"

MultiGaussianState<5>
MultiGaussianStateTransform::outerMultiState (const reco::GsfTrack& tk)
{
  const reco::GsfTrackExtraRef& extra(tk.gsfExtra());
  return multiState(extra->outerStateLocalParameters(),
		    extra->outerStateCovariances(),
		    extra->outerStateWeights());
}

MultiGaussianState<5>
MultiGaussianStateTransform::innerMultiState (const reco::GsfTrack& tk)
{
  const reco::GsfTrackExtraRef& extra(tk.gsfExtra());
  return multiState(extra->innerStateLocalParameters(),
	       extra->innerStateCovariances(),
	       extra->innerStateWeights());
}

MultiGaussianState<5> 
MultiGaussianStateTransform::multiState (const std::vector<MultiGaussianState<5>::Vector>& parameters,
					 const std::vector<MultiGaussianState<5>::Matrix>& covariances,
					 const std::vector<double>& weights)
{
  unsigned int nc = parameters.size();
  MultiGaussianState<5>::SingleStateContainer components;
  components.reserve(nc);
  for ( unsigned int i=0; i<nc; ++i ) {
    MultiGaussianState<5>::SingleStatePtr 
      sgs(new MultiGaussianState<5>::SingleState(parameters[i],covariances[i],weights[i]));
    components.push_back(sgs);
  }
  return MultiGaussianState<5>(components);
}
