#include "TrackingTools/GsfTracking/interface/GsfChi2MeasurementEstimator.h"
#include "TrackingTools/GsfTools/interface/GetComponents.h"

#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
#include "TrackingTools/GsfTracking/interface/PosteriorWeightsCalculator.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

std::pair<bool, double> GsfChi2MeasurementEstimator::estimate(const TrajectoryStateOnSurface& tsos,
                                                              const TrackingRecHit& hit) const {
  GetComponents comps(tsos);
  auto const& tsvec = comps();
  if (tsvec.empty()) {
    edm::LogError("GsfChi2MeasurementEstimator") << "Trying to calculate chi2 of hit with respect to empty mixture!";
    return std::make_pair(false, 0.);
  }

  auto const& weights = PosteriorWeightsCalculator(tsvec).weights(hit);
  if (weights.empty())
    return std::make_pair(false, 0);

  //   Chi2MeasurementEstimator est(chiSquaredCut());
  double chi2 = 0.;
  int i = 0;
  for (auto const& ts : tsvec)
    chi2 += weights[i++] * theEstimator.estimate(ts, hit).second;
  // Done - normalisation of weights is ensured
  // by PosteriorWeightsCalculator
  return returnIt(chi2);
}
