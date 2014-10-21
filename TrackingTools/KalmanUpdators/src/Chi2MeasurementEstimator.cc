#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
#include "TrackingTools/KalmanUpdators/interface/Chi2MeasurementEstimator.h"
#include "TrackingTools/PatternTools/interface/MeasurementExtractor.h"
#include "DataFormats/TrackingRecHit/interface/KfComponentsHolder.h"
#include "DataFormats/GeometrySurface/interface/Plane.h"
#include "DataFormats/Math/interface/invertPosDefMatrix.h"

std::pair<bool,double> 
Chi2MeasurementEstimator::estimate(const TrajectoryStateOnSurface& tsos,
				   const TrackingRecHit& aRecHit) const {
    switch (aRecHit.dimension()) {
        case 1: return estimate<1>(tsos,aRecHit);
        case 2: return estimate<2>(tsos,aRecHit);
        case 3: return estimate<3>(tsos,aRecHit);
        case 4: return estimate<4>(tsos,aRecHit);
        case 5: return estimate<5>(tsos,aRecHit);
    }
    throw cms::Exception("RecHit of invalid size (not 1,2,3,4,5)");
}

template <unsigned int D> std::pair<bool,double> 
Chi2MeasurementEstimator::estimate(const TrajectoryStateOnSurface& tsos,
				   const TrackingRecHit& aRecHit) const {
  typedef typename AlgebraicROOTObject<D,5>::Matrix MatD5;
  typedef typename AlgebraicROOTObject<5,D>::Matrix Mat5D;
  typedef typename AlgebraicROOTObject<D,D>::SymMatrix SMatDD;
  typedef typename AlgebraicROOTObject<D>::Vector VecD;

  VecD r, rMeas; SMatDD R, RMeas; 
  MatD5 dummyProjMatrix;
  ProjectMatrix<double,5,D> dummyProjFunc;
  auto && v = tsos.localParameters().vector();
  auto && m = tsos.localError().matrix();
  KfComponentsHolder holder;
  holder.template setup<D>(&r, &R, &dummyProjMatrix, &dummyProjFunc, &rMeas, &RMeas, v, m);
  aRecHit.getKfComponents(holder);
 
  R += RMeas;
  invertPosDefMatrix(R);
  double est = ROOT::Math::Similarity(r - rMeas, R);
  return returnIt(est);
}
