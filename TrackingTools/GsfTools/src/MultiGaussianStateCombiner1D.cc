#include "TrackingTools/GsfTools/interface/MultiGaussianStateCombiner1D.h"

#include "FWCore/Utilities/interface/Exception.h"

#include <cfloat>

SingleGaussianState1D MultiGaussianStateCombiner1D::combine(const MultiGaussianState1D& theState) const {
  return combine(theState.components());
}

SingleGaussianState1D MultiGaussianStateCombiner1D::combine(const VSC& theComponents) const {
  if (theComponents.empty()) {
    throw cms::Exception("LogicError") << "MultiGaussianStateCombiner1D:: state container to collapse is empty";
    //     return SingleState(SingleState::Vector(),SingleState::Matrix(),0.);
    //     return SingleState(SingleState::Vector(),
    // 		       SingleState::Matrix(),0.);
    return SingleGaussianState1D();
  }

  const SingleGaussianState1D& firstState(theComponents.front());
  if (theComponents.size() == 1)
    return firstState;

  //   int size = firstState.mean().num_row();
  double meanMean(0.);
  double weightSum(0.);
  //   double weight;
  double measCovar1(0.);
  double measCovar2(0.);
  for (VSC::const_iterator mixtureIter1 = theComponents.begin(); mixtureIter1 != theComponents.end(); mixtureIter1++) {
    double weight = mixtureIter1->weight();
    weightSum += weight;

    double mean1 = mixtureIter1->mean();
    meanMean += weight * mean1;
    measCovar1 += weight * mixtureIter1->variance();

    for (VSC::const_iterator mixtureIter2 = mixtureIter1 + 1; mixtureIter2 != theComponents.end(); mixtureIter2++) {
      double posDiff = mean1 - mixtureIter2->mean();
      //       SingleState::Matrix s(1,1); //stupid trick to make CLHEP work decently
      //       measCovar2 +=weight * (*mixtureIter2).weight() *
      //       				s.similarity(posDiff.T().T());
      //       SingleState::Matrix mat;
      //       for ( unsigned int i1=0; i1<N; i1++ ) {
      // 	for ( unsigned int i2=0; i2<=i1; i2++ )  mat(i1,i2) = posDiff(i1)*posDiff(i2);
      //       }
      //       measCovar2 += weight * (*mixtureIter2).weight() * mat;
      //
      // TensorProd yields a general matrix - need to convert to a symm. matrix
      double covGen = posDiff * posDiff;
      //       double covSym(covGen.LowerBlock());
      measCovar2 += weight * mixtureIter2->weight() * covGen;
    }
  }

  double measCovar;
  if (weightSum < DBL_MIN) {
    std::cout << "MultiGaussianStateCombiner1D:: New state has total weight of 0." << std::endl;
    //     meanMean = SingleState::Vector(size,0);
    meanMean = 0.;
    measCovar = 0.;
    weightSum = 0.;
  } else {
    weightSum = 1. / weightSum;
    meanMean *= weightSum;
    measCovar1 *= weightSum;
    measCovar2 *= weightSum * weightSum;
    measCovar = measCovar1 + measCovar2;
    //     measCovar = measCovar1/weightSum + measCovar2/weightSum/weightSum;
  }

  return SingleGaussianState1D(meanMean, measCovar, weightSum);
}
