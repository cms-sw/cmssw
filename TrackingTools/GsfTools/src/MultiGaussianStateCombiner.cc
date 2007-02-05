#include "TrackingTools/GsfTools/interface/MultiGaussianStateCombiner.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"
#include <cfloat>


RCSingleGaussianState
MultiGaussianStateCombiner::combine(const RCMultiGaussianState & theState) const
{
  return combine(theState->components());
}

RCSingleGaussianState 
MultiGaussianStateCombiner::combine(const VSC& theComponents) const
{
  if (theComponents.empty()) {
    throw cms::Exception("LogicError")
      << "MultiGaussianStateCombiner:: state container to collapse is empty";
  }

  if (theComponents.size()==1) {
    return theComponents.front();
  }


  int size = theComponents.front()->mean().num_row();
  AlgebraicVector meanMean(size,0);
  double weightSum = 0.;
  double weight;
  AlgebraicSymMatrix measCovar1(size, 0), measCovar2(size, 0);
  for (VSC::const_iterator mixtureIter1 = theComponents.begin();
  	mixtureIter1 != theComponents.end(); mixtureIter1++ ) {
    weight = (**mixtureIter1).weight();
    weightSum += weight;

    AlgebraicVector mean1 = (**mixtureIter1).mean();
    meanMean += weight * mean1;
    measCovar1 += weight * (**mixtureIter1).covariance();

    for (VSC::const_iterator mixtureIter2 = mixtureIter1+1;
  	mixtureIter2 != theComponents.end(); mixtureIter2++ ) {
      AlgebraicVector posDiff = mean1 - (**mixtureIter2).mean();
      AlgebraicSymMatrix s(1,1); //stupid trick to make CLHEP work decently
      measCovar2 +=weight * (**mixtureIter2).weight() *
      				s.similarity(posDiff.T().T());
    }
  }

  AlgebraicSymMatrix measCovar(size,0);
  if (weightSum<DBL_MIN){
    edm::LogInfo("MultiGaussianStateCombiner") 
      << "MultiGaussianStateCombiner:: New state has total weight of 0.\n";
    meanMean = AlgebraicVector(size,0);
    weightSum = 0.;
  } else {
    meanMean /= weightSum;
    measCovar = measCovar1/weightSum + measCovar2/weightSum/weightSum;
  }

// ThS: Have to take here the one of the state of the original TSOS to produce
// the final state. I hope that it does have all the info, otherwise it
// could not be used from MultiGaussianState directly...
  return theComponents.front()->createNewState(meanMean, measCovar, weightSum);

}
