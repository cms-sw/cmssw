#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
#include "TrackingTools/KalmanUpdators/interface/Chi2MeasurementEstimator.h"
#include "TrackingTools/PatternTools/interface/MeasurementExtractor.h"
#include "DataFormats/TrackingRecHit/interface/KfComponentsHolder.h"
#include "DataFormats/GeometrySurface/interface/Plane.h"
#include "DataFormats/Math/interface/invertPosDefMatrix.h"

namespace {
  template <unsigned int D>
  double lestimate(const TrajectoryStateOnSurface& tsos, const TrackingRecHit& aRecHit) {
    typedef typename AlgebraicROOTObject<D, 5>::Matrix MatD5;
    typedef typename AlgebraicROOTObject<5, D>::Matrix Mat5D;
    typedef typename AlgebraicROOTObject<D, D>::SymMatrix SMatDD;
    typedef typename AlgebraicROOTObject<D>::Vector VecD;
    using ROOT::Math::SMatrixNoInit;

    VecD r, rMeas;
    SMatDD R(SMatrixNoInit{}), RMeas(SMatrixNoInit{});
    ProjectMatrix<double, 5, D> dummyProjFunc;
    auto&& v = tsos.localParameters().vector();
    auto&& m = tsos.localError().matrix();
    KfComponentsHolder holder;
    holder.template setup<D>(&r, &R, &dummyProjFunc, &rMeas, &RMeas, v, m);
    aRecHit.getKfComponents(holder);

    R += RMeas;
    invertPosDefMatrix(R);
    return ROOT::Math::Similarity(r - rMeas, R);
  }
}  // namespace

std::pair<bool, double> Chi2MeasurementEstimator::estimate(const TrajectoryStateOnSurface& tsos,
                                                           const TrackingRecHit& aRecHit) const {
  switch (aRecHit.dimension()) {
    case 1:
      return returnIt(lestimate<1>(tsos, aRecHit));
    case 2:
      return returnIt(lestimate<2>(tsos, aRecHit));
    case 3:
      return returnIt(lestimate<3>(tsos, aRecHit));
    case 4:
      return returnIt(lestimate<4>(tsos, aRecHit));
    case 5:
      return returnIt(lestimate<5>(tsos, aRecHit));
  }
  throw cms::Exception("RecHit of invalid size (not 1,2,3,4,5)");
}
