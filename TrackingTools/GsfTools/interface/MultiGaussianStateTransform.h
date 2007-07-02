#ifndef MultiGaussianStateTransform_H
#define MultiGaussianStateTransform_H

/** Extracts innermost and  outermost states from a GsfTrack
 *  in form of a MultiGaussianState */

#include "DataFormats/GsfTrackReco/interface/GsfTrack.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrackExtra.h"

#include "TrackingTools/GsfTools/interface/MultiGaussianState.h"

#include <vector>

namespace MultiGaussianStateTransform {

  /** Construct a MultiGaussianState from the reco::GsfTrack 
   *  innermost state (local parameters) */
  MultiGaussianState<5> innerMultiState ( const reco::GsfTrack& tk);
  /** Construct a MultiGaussianState from the reco::GsfTrack 
   *  innermost state (local parameters) */
  MultiGaussianState<5> outerMultiState ( const reco::GsfTrack& tk);

  /** Construct a MultiGaussianState from the vectors of parameters,
   *  covariances and weights */
  MultiGaussianState<5> multiState (const std::vector<MultiGaussianState<5>::Vector>&,
				    const std::vector<MultiGaussianState<5>::Matrix>&,
				    const std::vector<double>&);
}

#endif
