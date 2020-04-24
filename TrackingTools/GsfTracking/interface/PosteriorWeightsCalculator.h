#ifndef PosteriorWeightsCalculator_H_
#define PosteriorWeightsCalculator_H_

#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
#include "TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHit.h"

/** Helper class which calculates the posterior weights of a Gaussian
 *  mixture given a prior (predicted) mixture and a RecHit. The prior
 *  is specified during construction time in the form of a vector of
 *  trajectory states.
 */

class PosteriorWeightsCalculator {

private:
  typedef TrajectoryStateOnSurface TSOS;

public:
  PosteriorWeightsCalculator(const std::vector<TSOS>& mixture) :
    predictedComponents(mixture) {}

  ~PosteriorWeightsCalculator() {}
  /// Create random state
  std::vector<double> weights(const TrackingRecHit& tsos) const;
  template <unsigned int D>
  std::vector<double> weights(const TrackingRecHit& tsos) const;

private:
  std::vector<TSOS> predictedComponents;
  
};

#endif //_TR_PosteriorWeightsCalculator_H_
