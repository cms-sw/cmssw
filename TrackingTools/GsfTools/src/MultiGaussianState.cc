#include "TrackingTools/GsfTools/interface/MultiGaussianState.h"
#include "TrackingTools/GsfTools/interface/MultiGaussianStateCombiner.h"

double MultiGaussianState::weight() const
{
  checkCombinedState();
  return theCombinedState->weight();
}

const AlgebraicVector & MultiGaussianState::mean() const
{
  checkCombinedState();
  return theCombinedState->mean();
}

const AlgebraicSymMatrix & MultiGaussianState::covariance() const
{
  checkCombinedState();
  return theCombinedState->covariance();
}

void MultiGaussianState::checkCombinedState() const
{
  if (theCombinedStateUp2Date) return;

  MultiGaussianStateCombiner theCombiner;
  theCombinedState = theCombiner.combine(theComponents);
  theCombinedStateUp2Date = true;

}
