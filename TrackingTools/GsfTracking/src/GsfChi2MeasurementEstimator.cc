#include "TrackingTools/GsfTracking/interface/GsfChi2MeasurementEstimator.h"

// #include "TrackingTools/KalmanUpdators/interface/Chi2MeasurementEstimator.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
#include "TrackingTools/GsfTracking/interface/PosteriorWeightsCalculator.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

std::pair<bool,double> 
GsfChi2MeasurementEstimator::estimate (const TrajectoryStateOnSurface& tsos,
				       const TrackingRecHit& hit) const {

  std::vector<TrajectoryStateOnSurface> tsvec = tsos.components();
  if (tsvec.empty()) {
    edm::LogError("GsfChi2MeasurementEstimator") 
      << "Trying to calculate chi2 of hit with respect to empty mixture!";
    return std::make_pair(false,0.);
  }

  std::vector<double> weights = PosteriorWeightsCalculator(tsvec).weights(hit);
  if ( weights.empty() )  return std::make_pair(false,0);

  //   Chi2MeasurementEstimator est(chiSquaredCut());
  double chi2 = 0.;
  int i = 0;
  for (std::vector<TrajectoryStateOnSurface>::const_iterator it = tsvec.begin();
       it != tsvec.end(); it++) {
    chi2 += weights[i++] * theEstimator.estimate(*it,hit).second;
  }
  // Done - normalisation of weights is ensured 
  // by PosteriorWeightsCalculator
  return returnIt(chi2);
}

