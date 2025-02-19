#ifndef MultiGaussianStateTransform_H
#define MultiGaussianStateTransform_H

/** Extracts innermost and  outermost states from a GsfTrack
 *  in form of a MultiGaussianState */

#include "DataFormats/GsfTrackReco/interface/GsfTrack.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrackExtra.h"

#include "TrackingTools/GsfTools/interface/MultiGaussianState.h"

#include <vector>

class MultiGaussianState1D;
class TrajectoryStateOnSurface;

namespace MultiGaussianStateTransform {

  enum { N = reco::GsfTrackExtra::dimension };


  /** Construct a MultiGaussianState from the reco::GsfTrack 
   *  innermost state (local parameters) */
  MultiGaussianState<N> innerMultiState ( const reco::GsfTrack& tk);
  /** Construct a MultiGaussianState from the reco::GsfTrack 
   *  innermost state (local parameters) */
  MultiGaussianState<N> outerMultiState ( const reco::GsfTrack& tk);
  /** Construct a MultiGaussianState1D for the local parameter corresponding
   *  to "index" (0<=index<5) from the reco::GsfTrack innermost state */
  MultiGaussianState1D innerMultiState1D ( const reco::GsfTrack& tk, unsigned int index);
  /** Construct a MultiGaussianState1D for the local parameter corresponding
   *  to "index" (0<=index<5) from the reco::GsfTrack outermost state */
  MultiGaussianState1D outerMultiState1D ( const reco::GsfTrack& tk, unsigned int index);

  /** Construct a MultiGaussianState from the vectors of parameters,
   *  covariances and weights */
  MultiGaussianState<N> multiState (const std::vector<MultiGaussianState<N>::Vector>&,
				    const std::vector<MultiGaussianState<N>::Matrix>&,
				    const std::vector<double>&);
  /** Construct a MultiGaussianState1D from the vectors of parameters,
   *  covariances and weights */
  MultiGaussianState1D multiState1D (const std::vector<MultiGaussianState<N>::Vector>&,
				     const std::vector<MultiGaussianState<N>::Matrix>&,
				     const std::vector<double>&,
				     unsigned int);

  /** Construct a MultiGaussianState from a TrajectoryStateOnSurface
   *  (local parameters) */
  MultiGaussianState<5> multiState (const TrajectoryStateOnSurface );
  /** Construct a MultiGaussianState1D from a TrajectoryStateOnSurface
   *  (local parameters) */
  MultiGaussianState1D multiState1D (const TrajectoryStateOnSurface,
				     unsigned int);
  /** Construct a TrajectoryStateOnSurface from a 5D SingleGaussianState
   *  (local parameters) and a reference TSOS (surface, charge, ..) */
  TrajectoryStateOnSurface tsosFromSingleState (const SingleGaussianState<5>&,
						const TrajectoryStateOnSurface);

}

#endif
