#include "TrackingTools/GsfTools/interface/MultiGaussianState1D.h"
#include "TrackingTools/GsfTools/interface/MultiGaussianStateCombiner1D.h"
// #include "TrackingTools/GsfTools/interface/MultiGaussianState.h"

double MultiGaussianState1D::weight() const
{
  if (theCombinedStateUp2Date) return theCombinedState.weight();
  
  double result(0.);
  for ( SingleState1dContainer::const_iterator ic=theComponents.begin();
	ic!=theComponents.end(); ic++ )  result += (*ic).weight();
  return result;
}

double MultiGaussianState1D::mean() const
{
  checkCombinedState();
  return theCombinedState.mean();
}

double MultiGaussianState1D::variance() const
{
  checkCombinedState();
  return theCombinedState.variance();
}

void MultiGaussianState1D::checkCombinedState() const
{
  if (theCombinedStateUp2Date) return;

  MultiGaussianStateCombiner1D combiner;
  theCombinedState = combiner.combine(theComponents);

//   typedef SingleGaussianState<1> SingleState;
//   typedef boost::shared_ptr< SingleGaussianState<1> > SingleStatePtr;
//   typedef std::vector< SingleStatePtr > SingleStateContainer;

//   SingleStateContainer components;
//   for ( SingleState1dContainer::const_iterator ic=theComponents.begin();
// 	ic!=theComponents.end(); ic++ ) {
//     SingleStatePtr ssp(new SingleState(SingleState::Vector((*ic).mean()),
// 				       SingleState::Matrix((*ic).variance()),
// 				       (*ic).weight()));
//     components.push_back(ssp);
//   }
//   MultiGaussianState<1> multiState(components);

//   theCombinedState = SingleGaussianState1D(multiState.mean()(0),
// 					   multiState.covariance()(0,0),
// 					   multiState.weight());

  theCombinedStateUp2Date = true;

}
