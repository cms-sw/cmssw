#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
#include "TrackingTools/KalmanUpdators/interface/Chi2MeasurementEstimator.h"
#include "TrackingTools/PatternTools/interface/MeasurementExtractor.h"
#include "DataFormats/GeometrySurface/interface/BoundPlane.h"

std::pair<bool,double> 
Chi2MeasurementEstimator::estimate(const TrajectoryStateOnSurface& tsos,
				   const TransientTrackingRecHit& aRecHit) const {
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
				   const TransientTrackingRecHit& aRecHit) const {
  typedef typename AlgebraicROOTObject<D>::Vector Vec;
  typedef typename AlgebraicROOTObject<D>::SymMatrix Mat;
  
  MeasurementExtractor me(tsos);
  Vec r = asSVector<D>(aRecHit.parameters()) - me.measuredParameters<D>(aRecHit);
  Mat R = asSMatrix<D>(aRecHit.parametersError()) + me.measuredError<D>(aRecHit);
  //int ierr = ! R.Invert(); // if (ierr != 0) throw exception; // 
  R.Invert();
  double est = ROOT::Math::Similarity(r, R);
  return returnIt(est);
}
