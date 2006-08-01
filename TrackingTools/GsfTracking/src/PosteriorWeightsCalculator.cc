#include "TrackingTools/GsfTracking/interface/PosteriorWeightsCalculator.h"

#include "TrackingTools/PatternTools/interface/MeasurementExtractor.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <cfloat>

std::vector<double> PosteriorWeightsCalculator::weights(const TransientTrackingRecHit& recHit) const {

  std::vector<double> weights;
  if ( predictedComponents.empty() )  return weights;
  weights.reserve(predictedComponents.size());

  std::vector<double> detRs;
  detRs.reserve(predictedComponents.size());
  std::vector<double> chi2s;
  chi2s.reserve(predictedComponents.size());
  //
  // calculate chi2 and determinant / component and find
  //   minimum / maximum of chi2
  //  
  double chi2Min(DBL_MAX);
  for ( unsigned int i=0; i<predictedComponents.size(); i++ ) {
    MeasurementExtractor me(predictedComponents[i]);
    // Residuals of aPredictedState w.r.t. aRecHit, 
    //!!!     AlgebraicVector r(recHit.parameters(predictedComponents[i]) - me.measuredParameters(recHit));
    AlgebraicVector r(recHit.parameters() - me.measuredParameters(recHit));
    // and covariance matrix of residuals
    //!!!     AlgebraicSymMatrix V(recHit.parametersError(predictedComponents[i]));
    AlgebraicSymMatrix V(recHit.parametersError());
    AlgebraicSymMatrix R(V + me.measuredError(recHit));
    double detR = R.determinant();
    detRs.push_back(detR);

    int ierr; R.invert(ierr); // if (ierr != 0) throw exception;
    if ( ierr!=0 )  
      edm::LogError("PosteriorWeightsCalculator") 
	<< "PosteriorWeightsCalculator: inversion failed, ierr = " << ierr;
    double chi2 = R.similarity(r);
    chi2s.push_back(chi2);
    if ( chi2<chi2Min )  chi2Min = chi2;
  }
  if ( detRs.size()!=predictedComponents.size() ||
       chi2s.size()!=predictedComponents.size() )  
    edm::LogError("PosteriorWeightsCalculator") << "Problem in vector sizes";
  //
  // calculate weights (extracting a common factor
  //   exp(-0.5*chi2Min) to avoid numerical problems
  //   during exponentation
  //
  double sumWeights(0.);
  for ( unsigned int i=0; i<predictedComponents.size(); i++ ) {
    double priorWeight = predictedComponents[i].weight();

    double chi2 = chi2s[i] - chi2Min;

    double tempWeight(0.);
    if ( detRs[i]>FLT_MIN ) {
      //
      // Calculation of (non-normalised) weight. Common factors exp(-chi2Norm/2.) and
      // 1./sqrt(2*pi*recHit.dimension()) have been omitted
      //
      tempWeight = priorWeight * sqrt(1./detRs[i]) * exp(-0.5 * chi2); 
    }
    //      else {
    //        edm::LogInfo("PosteriorWeightsCalculator") << "PosteriorWeightsCalculator: detR < FLT_MIN !!";
    //      }
    weights.push_back(tempWeight);
    sumWeights += tempWeight;
  }
  if ( sumWeights<DBL_MIN ) {
    edm::LogError("PosteriorWeightsCalculator") << "PosteriorWeightsCalculator: sumWeight < DBL_MIN";
    return std::vector<double>();
  }

  if ( weights.size()!=predictedComponents.size() )  
    edm::LogError("PosteriorWeightsCalculator") << "Problem in vector sizes (2)";
  for (std::vector<double>::iterator iter = weights.begin();
       iter != weights.end(); iter++) {
    (*iter) /= sumWeights;
  }

  return weights;
}
