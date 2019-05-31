#include "TrackingTools/GsfTracking/interface/PosteriorWeightsCalculator.h"

#include "TrackingTools/PatternTools/interface/MeasurementExtractor.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/TrackingRecHit/interface/KfComponentsHolder.h"
#include "DataFormats/Math/interface/invertPosDefMatrix.h"
#include "DataFormats/Math/interface/ProjectMatrix.h"

#include <cfloat>

std::vector<double> PosteriorWeightsCalculator::weights(const TrackingRecHit& recHit) const {
  switch (recHit.dimension()) {
    case 1:
      return weights<1>(recHit);
    case 2:
      return weights<2>(recHit);
    case 3:
      return weights<3>(recHit);
    case 4:
      return weights<4>(recHit);
    case 5:
      return weights<5>(recHit);
  }
  throw cms::Exception("Error: rechit of size not 1,2,3,4,5");
}

template <unsigned int D>
std::vector<double> PosteriorWeightsCalculator::weights(const TrackingRecHit& recHit) const {
  typedef typename AlgebraicROOTObject<D, 5>::Matrix MatD5;
  typedef typename AlgebraicROOTObject<5, D>::Matrix Mat5D;
  typedef typename AlgebraicROOTObject<D, D>::SymMatrix SMatDD;
  typedef typename AlgebraicROOTObject<D>::Vector VecD;
  using ROOT::Math::SMatrixNoInit;

  std::vector<double> weights;
  if (predictedComponents.empty()) {
    edm::LogError("EmptyPredictedComponents") << "a multi state is empty. cannot compute any weight.";
    return weights;
  }
  weights.reserve(predictedComponents.size());

  std::vector<double> detRs;
  detRs.reserve(predictedComponents.size());
  std::vector<double> chi2s;
  chi2s.reserve(predictedComponents.size());

  VecD r, rMeas;
  SMatDD V(SMatrixNoInit{}), R(SMatrixNoInit{});
  ProjectMatrix<double, 5, D> p;
  //
  // calculate chi2 and determinant / component and find
  //   minimum / maximum of chi2
  //
  double chi2Min(DBL_MAX);
  for (unsigned int i = 0; i < predictedComponents.size(); i++) {
    KfComponentsHolder holder;
    auto const& x = predictedComponents[i].localParameters().vector();
    holder.template setup<D>(&r, &V, &p, &rMeas, &R, x, predictedComponents[i].localError().matrix());
    recHit.getKfComponents(holder);

    r -= rMeas;
    R += V;

    double detR;
    if (!R.Det2(detR)) {
      edm::LogError("PosteriorWeightsCalculator") << "PosteriorWeightsCalculator: determinant failed";
      return std::vector<double>();
    }
    detRs.push_back(detR);

    bool ok = invertPosDefMatrix(R);
    if (!ok) {
      edm::LogError("PosteriorWeightsCalculator") << "PosteriorWeightsCalculator: inversion failed";
      return std::vector<double>();
    }
    double chi2 = ROOT::Math::Similarity(r, R);
    chi2s.push_back(chi2);
    if (chi2 < chi2Min)
      chi2Min = chi2;
  }

  if (detRs.size() != predictedComponents.size() || chi2s.size() != predictedComponents.size()) {
    edm::LogError("PosteriorWeightsCalculator") << "Problem in vector sizes";
    return std::vector<double>();
  }

  //
  // calculate weights (extracting a common factor
  //   exp(-0.5*chi2Min) to avoid numerical problems
  //   during exponentation
  //
  double sumWeights(0.);
  for (unsigned int i = 0; i < predictedComponents.size(); i++) {
    double priorWeight = predictedComponents[i].weight();

    double chi2 = chi2s[i] - chi2Min;

    double tempWeight(0.);
    if (detRs[i] > FLT_MIN) {
      //
      // Calculation of (non-normalised) weight. Common factors exp(-chi2Norm/2.) and
      // 1./sqrt(2*pi*recHit.dimension()) have been omitted
      //
      tempWeight = priorWeight * std::sqrt(1. / detRs[i]) * std::exp(-0.5 * chi2);
    } else {
      LogDebug("GsfTrackFitters") << "PosteriorWeightsCalculator: detR < FLT_MIN !!";
    }
    weights.push_back(tempWeight);
    sumWeights += tempWeight;
  }

  if (sumWeights < DBL_MIN) {
    LogDebug("GsfTrackFitters") << "PosteriorWeightsCalculator: sumWeight < DBL_MIN";
    edm::LogError("PosteriorWeightsCalculator") << "PosteriorWeightsCalculator: sumWeight < DBL_MIN";
    return std::vector<double>();
  }

  if (weights.size() != predictedComponents.size()) {
    edm::LogError("PosteriorWeightsCalculator") << "Problem in vector sizes (2)";
    return std::vector<double>();
  }
  sumWeights = 1. / sumWeights;
  for (auto& w : weights)
    w *= sumWeights;
  return weights;
}
